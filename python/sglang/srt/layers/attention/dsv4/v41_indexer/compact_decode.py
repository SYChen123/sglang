"""Hopper candidate consumers score compact logical block lists.

The source still scores its visible context. It publishes sorted blocks and a
valid prefix length once; later layers score only that prefix. Shapes/chunks
depend on the replay capacity, never a CPU read of the current sequence length.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.attention.deep_select import get_input_stride_alignment_bytes
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_paged
from sglang.kernels.ops.attention.dsv4.sm90_candidate import (
    candidate_block_state,
    candidate_position_counts,
    finalize_candidate_topk,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.srt.environ import envs

from .types import CandidateMetadata, DecodeInputs

_SCORE_BUDGET_BYTES = 512 << 20


def _score_rows_per_chunk(width: int, *, block_size: int | None) -> int:
    # Account for the allocations made by fp4_index_logits_paged: FP32 logits
    # aligned for DeepSelect, plus aligned FP32 block maxima for the source.
    alignment = get_input_stride_alignment_bytes()
    row_bytes = (width * 4 + alignment - 1) // alignment * alignment
    if block_size is not None:
        blocks = (width + block_size - 1) // block_size
        row_bytes += (blocks * 4 + 15) // 16 * 16
    # Leave room for selection indices, sorting and TopK plans. The old 8 bytes
    # per score substantially overestimated these per-chunk allocations.
    reserve = min(8 << 20, _SCORE_BUDGET_BYTES // 16)
    return max(1, (_SCORE_BUDGET_BYTES - reserve) // max(1, row_bytes))


class CompactDecodeBlocks(CandidateMetadata, msgspec.Struct):
    ratio: int
    block_size: int
    blocks: torch.Tensor
    counts: torch.Tensor
    topk_plan: torch.Tensor


def decode_capacity(inputs: DecodeInputs, req_to_token: torch.Tensor) -> int:
    return min(
        inputs.paged_metadata.max_compressed_seq_len,
        req_to_token.shape[1] // inputs.compress_ratio,
    )


def can_publish_compact(inputs, req_to_token, *, topk_blocks, block_size) -> bool:
    return (
        block_size in (1, 2, 4, 8, 16, 32, 64)
        and 0 < topk_blocks <= 2048
        and inputs.indexer.index_topk <= 2048
        and decode_capacity(inputs, req_to_token) >= 16 * topk_blocks * block_size
    )


def compact_decode(
    inputs: DecodeInputs,
    *,
    pool,
    req_to_token: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    published: CompactDecodeBlocks | None = None,
) -> CompactDecodeBlocks | None:
    inputs.reset_outputs()
    req, pos = inputs.req_rows, inputs.positions
    bs, ratio = req.shape[0], inputs.compress_ratio
    assert pos.shape[0] == bs
    lmax = decode_capacity(inputs, req_to_token)
    if bs == 0 or lmax == 0:
        return None
    indexer = inputs.indexer
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
    lens = ((pos + 1) // ratio).clamp(0, lmax).to(torch.int32)
    blocks, score_lens, shared_plan = None, lens, None
    if published is not None:
        assert published.blocks.shape[0] == bs, "candidate query mapping changed"
        assert published.block_size == block_size
        blocks = published.blocks
        score_lens = published.counts
        if ratio == published.ratio:
            shared_plan = published.topk_plan
        else:
            score_lens = candidate_position_counts(
                blocks, published.counts, lens, block_size=block_size
            )
    width = lmax if blocks is None else blocks.shape[1] * block_size
    rows_per_chunk = _score_rows_per_chunk(
        width, block_size=block_size if published is None else None
    )
    use_fp8 = envs.SGLANG_OPT_DSV41_DECODE_INDEXER_FP8.get()
    k_reuse_group_size = (
        inputs.k_reuse_group_size if published is None and use_fp8 else 1
    )
    if k_reuse_group_size == -1 and bs > rows_per_chunk:
        # Keep request reuse if the existing chunks preserve complete requests.
        verify_width = inputs.verify_tokens_per_request
        if rows_per_chunk % verify_width:
            k_reuse_group_size = 2 if verify_width % 2 == 0 else 1
    if k_reuse_group_size == 2 and rows_per_chunk > 1:
        rows_per_chunk -= rows_per_chunk % 2
    single_chunk = bs <= rows_per_chunk
    source_blocks = source_counts = None
    if published is None and not single_chunk:
        source_blocks = torch.empty(
            (bs, topk_blocks), dtype=torch.int32, device=pos.device
        )
        source_counts = torch.empty(bs, dtype=torch.int32, device=pos.device)
    k = min(indexer.index_topk, width)
    for start in range(0, bs, rows_per_chunk):
        rows = slice(start, min(start + rows_per_chunk, bs))
        lengths = score_lens[rows]
        scores = fp4_index_logits_paged(
            q[rows],
            weights[rows],
            req[rows],
            req_to_token,
            lengths,
            table,
            table.shape[1] // 68,
            lmax,
            ratio,
            candidate_blocks=blocks[rows] if blocks is not None else None,
            candidate_block_size=block_size,
            return_block_scores=published is None,
            use_fp8=use_fp8,
            k_reuse_group_size=k_reuse_group_size,
            verify_tokens_per_request=inputs.verify_tokens_per_request,
        )
        if published is None:
            scores, block_scores = scores
            selected_blocks, counts = candidate_block_state(
                block_scores, lengths, topk_blocks=topk_blocks, block_size=block_size
            )
            if single_chunk:
                source_blocks, source_counts = selected_blocks, counts
            else:
                source_blocks[rows] = selected_blocks
                source_counts[rows] = counts
            del block_scores, selected_blocks, counts
        selected = torch.empty(
            (q[rows].shape[0], k), dtype=torch.int32, device=q.device
        )
        plan = shared_plan if single_chunk else None
        if plan is None:
            plan = plan_topk_v2(lengths)
        topk_transform_paged_v2(scores, lengths, None, selected, 1, plan)
        finalize_candidate_topk(
            selected,
            scores,
            lengths,
            req_to_token,
            req[rows],
            inputs.out_page_indices[rows],
            None,
            ratio=ratio,
            candidate_blocks=blocks[rows] if blocks is not None else None,
            candidate_block_size=block_size,
        )
        del scores, selected
    if published is None:
        return CompactDecodeBlocks(
            ratio=ratio,
            block_size=block_size,
            blocks=source_blocks,
            counts=source_counts,
            topk_plan=plan_topk_v2(source_counts),
        )
    return None
