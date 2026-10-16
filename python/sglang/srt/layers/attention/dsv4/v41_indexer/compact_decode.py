"""Hopper candidate consumers score compact logical block lists.

The source still scores its visible context. It publishes sorted blocks and a
valid prefix length once; later layers score only that prefix. Shapes/chunks
depend on the replay capacity, never a CPU read of the current sequence length.
"""

from __future__ import annotations

import msgspec
import torch

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

from .types import CandidateMetadata, DecodeInputs

_SCORE_BUDGET_BYTES = 64 << 20


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
    rows_per_chunk = max(1, _SCORE_BUDGET_BYTES // (width * 8))
    source_blocks = source_counts = None
    if published is None:
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
        )
        if published is None:
            scores, block_scores = scores
            selected_blocks, counts = candidate_block_state(
                block_scores, lengths, topk_blocks=topk_blocks, block_size=block_size
            )
            source_blocks[rows] = selected_blocks
            source_counts[rows] = counts
            del block_scores, selected_blocks, counts
        selected = torch.empty(
            (q[rows].shape[0], k), dtype=torch.int32, device=q.device
        )
        plan = shared_plan if bs <= rows_per_chunk else None
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
