# SPDX-License-Identifier: Apache-2.0
"""Capture-stable interleave CP buffers for decoder SWA bounded replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata

_DENSE_TAIL_WINDOWS = 8
_TAIL_PADDING_DIVISOR = 4


def required_tail_rows_per_rank(extend_lens, *, window: int, cp_size: int) -> int:
    """Largest rank share of tails in the original interleave order.

    Dividing the aggregate by CP misses short tails with the same owner.
    CPU-only selection also avoids a GPU sync on every graph dispatch.
    """
    counts = [0] * cp_size
    end = 0
    for length in extend_lens:
        end += int(length)
        start = end - min(window, int(length))
        for rank in range(cp_size):
            counts[rank] += (end + cp_size - 1 - rank) // cp_size - (
                start + cp_size - 1 - rank
            ) // cp_size
    return max(counts)


def capture_tail_buckets_per_rank(
    capacity: int, *, window: int, cp_size: int
) -> tuple[int, ...]:
    """Use one-window spacing for small tails and bounded growth after that."""
    step = (window + cp_size - 1) // cp_size
    buckets = []
    rows = step
    while rows < capacity:
        buckets.append(rows)
        windows = rows // step
        rows += step * (
            1 if windows < _DENSE_TAIL_WINDOWS else windows // _TAIL_PADDING_DIVISOR
        )
    buckets.append(capacity)
    return tuple(buckets)


def select_tail_bucket(
    buckets: tuple[int, ...], required: int, *, window: int, cp_size: int
) -> int | None:
    step = (window + cp_size - 1) // cp_size
    # Allow rounding to the next window for short tails. Large tails have a
    # relative limit, including when the smallest capture was filtered out.
    max_padding = max(step - 1, required // _TAIL_PADDING_DIVISOR)
    return next(
        (rows for rows in buckets if required <= rows <= required + max_padding),
        None,
    )


@dataclass
class CPTailGraphBuffers:
    token_indices: torch.Tensor
    positions: torch.Tensor
    scatter_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    swa_out_cache_loc: torch.Tensor
    req_global: torch.Tensor
    pos_global: torch.Tensor
    gather_index: torch.Tensor
    num_token_non_padded: torch.Tensor

    @classmethod
    def create(cls, forward_batch, *, local_rows: int, cp_size: int):
        device = forward_batch.input_ids.device

        def zeros(rows, dtype):
            return torch.zeros(rows, dtype=dtype, device=device)

        global_rows = local_rows * cp_size
        return cls(
            token_indices=zeros(local_rows, torch.int64),
            positions=zeros(local_rows, forward_batch.positions.dtype),
            scatter_indices=zeros(local_rows, torch.int64),
            out_cache_loc=zeros(global_rows, forward_batch.out_cache_loc.dtype),
            swa_out_cache_loc=zeros(global_rows, torch.int32),
            req_global=zeros(global_rows, torch.int64),
            pos_global=zeros(global_rows, torch.int64),
            gather_index=zeros(global_rows, torch.int64),
            num_token_non_padded=torch.zeros((), dtype=torch.int32, device=device),
        )

    def update(self, forward_batch, global_indices, tail_lens, layout, *, cp_rank):
        """Real rows precede padding in both sequence and rank-local order.

        Graph segments always see the same physical shapes and addresses. The
        gather permutation is refreshed before replay; the eager logits tail
        drops its padded suffix after the collective.
        """
        metadata = layout["cp_metadata"]
        cp_size = len(metadata.per_rank_actual_token)
        local_rows = self.token_indices.numel()
        real_local = sum(layout["local_lens_cpu"])
        real_global = global_indices.numel()
        assert real_local <= local_rows
        assert real_global <= self.gather_index.numel()
        self.token_indices.zero_()
        self.token_indices[:real_local].copy_(layout["local_token_indices"])
        self.positions.copy_(layout["local_positions"])
        full_local_rows = forward_batch.attn_cp_metadata.per_rank_actual_token[cp_rank]
        self.scatter_indices.fill_(full_local_rows)
        self.scatter_indices[:real_local].copy_(layout["local_token_indices"])
        self.num_token_non_padded.fill_(real_local)
        self.out_cache_loc.zero_()
        self.pos_global.zero_()
        self.req_global.zero_()
        self.out_cache_loc[:real_global].copy_(
            forward_batch.out_cache_loc[global_indices]
        )
        self.pos_global[:real_global].copy_(forward_batch.positions[global_indices])
        self.req_global[:real_global].copy_(
            torch.repeat_interleave(
                forward_batch.req_pool_indices.to(torch.int64),
                tail_lens.to(torch.int64),
                output_size=real_global,
            )
        )
        self.gather_index[:real_global].copy_(metadata.gather_index)
        cursor = real_global
        for rank, count in enumerate(metadata.per_rank_logical_token):
            padding = local_rows - count
            self.gather_index[cursor : cursor + padding].copy_(
                torch.arange(
                    rank * local_rows + count,
                    (rank + 1) * local_rows,
                    device=self.gather_index.device,
                )
            )
            cursor += padding
        return InterleaveContextParallelMetadata(
            per_rank_actual_token=[local_rows] * cp_size,
            max_rank_len=[local_rows] * cp_size,
            per_rank_logical_token=[local_rows] * cp_size,
            total_seq_lens=local_rows * cp_size,
            bs=forward_batch.batch_size,
            gather_index=self.gather_index,
            local_index=metadata.local_index,
            moe_local_token_count=self.num_token_non_padded,
        )
