"""Hopper eager prefill: gather each request's FP4 K once, then score in FP8.

CP owns a consecutive group of rank-local rows per request. Use those counts,
including empty shards, rather than synchronizing lengths from a CUDA tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from .types import PrefillInputs

# Only the final [query, key] FP32 scores are materialized. Leave headroom for
# candidate selection and top-k scratch; this does not change model prefill size.
_SCORE_BUDGET_BYTES = 64 << 20


def use_sm90_prefill(inputs: PrefillInputs) -> bool:
    return (
        inputs.x.is_cuda
        and get_platform().is_sm90
        and not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
        and inputs.x.dtype == torch.bfloat16
        and inputs.indexer.index_head_dim == 128
        and inputs.indexer.n_local_heads in (32, 64)
        and inputs.seq_lens_cpu is not None
        and inputs.rows_per_request is not None
    )


def prefill_requests(
    inputs,
    token_to_kv_pool,
    req_to_token,
    *,
    candidate_blocks=None,
    candidate_block_size=8,
):
    from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
        fp8_index_logits_prefill,
        quantize_bf16_index_queries_fp8,
        unpack_fp4_index_keys_to_fp8,
    )

    from .scoring import ChunkScores, RequestScores

    backend = envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.get()
    if backend not in ("triton", "deepgemm", "auto"):
        raise ValueError(
            "SGLANG_DSV41_SM90_PREFILL_INDEXER must be 'triton', 'deepgemm' or 'auto'"
        )
    min_deepgemm_rows = envs.SGLANG_DSV41_SM90_PREFILL_INDEXER_DEEPGEMM_MIN_ROWS.get()
    if backend == "auto" and min_deepgemm_rows <= 0:
        raise ValueError(
            "SGLANG_DSV41_SM90_PREFILL_INDEXER_DEEPGEMM_MIN_ROWS must be positive"
        )
    if backend != "triton":
        from sglang.kernels.ops.attention.dsv4.sm90_deepgemm_prefill import (
            deepgemm_fp8_index_logits_prefill,
        )

    inputs.reset_outputs()
    counts, seq_lens = inputs.rows_per_request, inputs.seq_lens_cpu
    assert len(counts) == len(seq_lens) == inputs.req_pool_indices.numel()
    real_rows = sum(counts)
    if candidate_blocks is not None:
        assert candidate_blocks.shape[0] >= real_rows, "candidate query mapping changed"
    if real_rows == 0:
        return
    assert real_rows <= inputs.positions.numel()
    positions = inputs.positions[:real_rows]
    q = inputs.indexer.queries(inputs.q_lora[:real_rows], inputs.freqs_cis[positions])
    weights = inputs.indexer.head_weights(inputs.x[:real_rows])
    lens = ((positions + 1) // inputs.compress_ratio).to(torch.int32)
    table = token_to_kv_pool.get_index_k_with_scale_buffer(inputs.layer_id)
    page_size = table.shape[1] // 68
    row_base = 0
    for b, (count, seq_len) in enumerate(zip(counts, seq_lens)):
        lc = seq_len // inputs.compress_ratio
        begin, row_base = row_base, row_base + count
        if count == 0 or lc == 0:
            continue
        columns = torch.arange(lc, device=positions.device)
        req = inputs.req_pool_indices[b : b + 1].to(torch.int64)
        slots = req_to_token[req, columns * inputs.compress_ratio].to(torch.int64)
        slots = slots // inputs.compress_ratio
        keys = unpack_fp4_index_keys_to_fp8(slots, table, page_size)
        tok = torch.arange(begin, row_base, device=positions.device)
        request = RequestScores(
            lc=lc,
            k=min(inputs.indexer.index_topk, lc),
            columns=columns,
            tok=tok,
            lens=lens[begin:row_base],
            slots=slots,
        )

        # Consume this iterator before advancing the outer request iterator.
        def chunks(request=request, keys=keys):
            # DG pads full-width cleaned output to ceil((N + 256) / 256)*256.
            # Auto uses the larger backing stride and DG row alignment so either
            # scorer fits the budget; choose only after the real tile is known.
            stride = (
                ((request.lc + 511) // 256) * 256
                if backend != "triton"
                else ((request.lc + 3) // 4) * 4
            )
            row_alignment = 128 // q.shape[1] if backend != "triton" else 1
            rows_per_chunk = max(
                row_alignment,
                (_SCORE_BUDGET_BYTES // (stride * 4)) // row_alignment * row_alignment,
            )
            key_scales = None
            for start in range(0, request.tok.numel(), rows_per_chunk):
                rows = slice(start, start + rows_per_chunk)
                token_rows = request.tok[rows]
                q_fp8 = quantize_bf16_index_queries_fp8(q[token_rows])
                use_deepgemm = backend == "deepgemm" or (
                    backend == "auto" and q_fp8.shape[0] >= min_deepgemm_rows
                )
                compact_blocks = None
                if use_deepgemm:
                    if key_scales is None:
                        # FP4 block exponents were applied during gather. DG
                        # wants one FP32 scale per K row, shared across tiles.
                        key_scales = torch.ones(
                            request.lc, device=keys.device, dtype=torch.float32
                        )
                    scores = deepgemm_fp8_index_logits_prefill(
                        q_fp8,
                        weights[token_rows],
                        keys,
                        request.lens[rows],
                        key_scales=key_scales,
                    )
                else:
                    if candidate_blocks is not None:
                        compact_blocks = candidate_blocks[token_rows]
                    scores = fp8_index_logits_prefill(
                        q_fp8,
                        weights[token_rows],
                        keys,
                        request.lens[rows],
                        candidate_blocks=compact_blocks,
                        candidate_block_size=candidate_block_size,
                    )
                yield ChunkScores(
                    rows=rows,
                    tok=token_rows,
                    lens=request.lens[rows],
                    scores=scores,
                    use_ragged_topk=True,
                    candidate_blocks=compact_blocks,
                )

        yield request, chunks()
