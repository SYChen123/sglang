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
):
    from sglang.kernels.ops.attention.dsv4.sm90_deepgemm_prefill import (
        deepgemm_fp8_index_logits_prefill,
    )
    from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
        quantize_bf16_index_queries_fp8,
        unpack_fp4_index_keys_to_fp8,
    )

    from .scoring import ChunkScores, RequestScores

    if envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.get() != "deepgemm":
        raise ValueError("SGLANG_DSV41_SM90_PREFILL_INDEXER must be 'deepgemm'")

    # Only the final [query, key] FP32 scores are materialized.
    score_budget_bytes = (
        2 << 30
        if envs.SGLANG_OPT_DSV41_LARGE_PREFILL_SCORE_BUDGET.get()
        else _SCORE_BUDGET_BYTES
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
            stride = ((request.lc + 511) // 256) * 256
            row_alignment = 128 // q.shape[1]
            rows_per_chunk = max(
                row_alignment,
                (score_budget_bytes // (stride * 4)) // row_alignment * row_alignment,
            )
            # The gather applies the FP4 exponents. DeepGEMM still expects
            # one scale per key row, shared across query tiles.
            key_scales = torch.ones(request.lc, device=keys.device, dtype=torch.float32)
            for start in range(0, request.tok.numel(), rows_per_chunk):
                rows = slice(start, start + rows_per_chunk)
                token_rows = request.tok[rows]
                q_fp8 = quantize_bf16_index_queries_fp8(q[token_rows])
                scores = deepgemm_fp8_index_logits_prefill(
                    q_fp8,
                    weights[token_rows],
                    keys,
                    request.lens[rows],
                    key_scales=key_scales,
                )
                yield ChunkScores(
                    rows=rows,
                    tok=token_rows,
                    lens=request.lens[rows],
                    scores=scores,
                    use_ragged_topk=True,
                )

        yield request, chunks()
