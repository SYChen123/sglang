"""Hopper V4.1 prefill scoring, adapted from sgl-project/sglang#39086.

Unpack visible FP4 K rows once, then fuse FP8 dot and BF16 post-dot rounding.
Keep this separate from the decode/verify compact candidate kernels.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_HEAD_DIM,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_K_PAYLOAD_BYTES as PAYLOAD_BYTES,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_K_SCALE_BYTES as SCALE_BYTES,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _e2m1_decode,
)

FP8_E4M3_MAX = 448.0


@triton.jit
def _unpack_fp4_index_keys_to_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Decode block-scaled E2M1 values directly into E4M3."""
    row = tl.program_id(0)
    offs_i = tl.arange(0, HALF_D)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(table_ptr + row_base + off * PAYLOAD_BYTES + offs_i)
    scale_block = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base
        + page_size * PAYLOAD_BYTES
        + off * SCALE_BYTES
        + scale_block
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    low = tl.clamp(_e2m1_decode(pay & 0x0F) * scale, -FP8_MAX, FP8_MAX).to(
        out_ptr.dtype.element_ty
    )
    high = tl.clamp(
        _e2m1_decode((pay >> 4) & 0x0F) * scale,
        -FP8_MAX,
        FP8_MAX,
    ).to(out_ptr.dtype.element_ty)
    out = out_ptr + row * (HALF_D * 2)
    tl.store(out + 2 * offs_i, low)
    tl.store(out + 2 * offs_i + 1, high)


def unpack_fp4_index_keys_to_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather FP4 index-K rows and decode them directly into E4M3."""
    assert slots.dim() == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert table.stride(1) == 1 and table.shape[1] == page_size * 68
    slots = slots.to(torch.int64).contiguous()
    values = torch.empty(
        (slots.shape[0], INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=slots.device,
    )
    if slots.numel() > 0:
        _unpack_fp4_index_keys_to_fp8_kernel[(slots.shape[0],)](
            slots,
            table,
            values,
            page_size,
            table.stride(0),
            HALF_D=INDEX_HEAD_DIM // 2,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=4,
        )
    return values


@triton.jit
def _quantize_bf16_index_queries_fp8_kernel(
    q_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :]
    ).to(tl.float32)
    q_fp8 = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :],
        q_fp8,
    )


def quantize_bf16_index_queries_fp8(
    q: torch.Tensor,
) -> torch.Tensor:
    """Cast each BF16 query head to E4M3 for one K=128 tensor-core dot."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    rows, heads, _ = q.shape
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if rows > 0:
        _quantize_bf16_index_queries_fp8_kernel[(rows,)](
            q,
            values,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            H=heads,
            D=INDEX_HEAD_DIM,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=8,
        )
    return values


@triton.jit
def _fp8_index_logits_prefill_kernel(
    q_ptr,  # [B, H, D] e4m3
    w_ptr,  # [B, H] bf16
    k_ptr,  # [L, D] e4m3
    lens_ptr,  # [B] int64
    blocks_ptr,  # optional [B, NUM_BLOCKS] int32 logical block ids
    out_ptr,  # [B, OUT_L] fp32
    L,
    OUT_L,
    stride_qb,
    stride_qh,
    stride_kl,
    stride_wb,
    stride_blocks,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    n_vis = tl.load(lens_ptr + b)
    logical = offs_l
    if USE_CANDIDATES:
        block = tl.load(
            blocks_ptr + b * stride_blocks + offs_l // CANDIDATE_BLOCK_SIZE,
            mask=offs_l < NUM_BLOCKS * CANDIDATE_BLOCK_SIZE,
            other=-1,
        )
        logical = block * CANDIDATE_BLOCK_SIZE + offs_l % CANDIDATE_BLOCK_SIZE
        valid = (block >= 0) & (logical < tl.minimum(n_vis, L))
    else:
        valid = offs_l < tl.minimum(n_vis, L)
    q = tl.load(q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :])
    k = tl.load(
        k_ptr + logical[:, None] * stride_kl + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
    # Preserve the reference post-dot rounding and reduction points.
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0, propagate_nan=tl.PropagateNan.ALL)
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * OUT_L + offs_l, logit, mask=offs_l < OUT_L)


def fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    *,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 8,
) -> torch.Tensor:
    """Score E4M3 queries against E4M3 keys with FP32 accumulation.

    Rows have a four-float-aligned storage stride for ragged top-k. The returned
    view keeps the logical width; backing padding is -inf. With candidates,
    column j scores K[blocks[row, j // block_size] * block_size + j % block_size].
    Each row owns its block list; holes and causally invisible positions are -inf.
    """
    assert q.dtype == torch.float8_e4m3fn and q.shape[-1] == INDEX_HEAD_DIM
    rows, heads, _ = q.shape
    assert heads in (32, 64)
    key_width = keys.shape[0]
    width = key_width
    assert keys.dtype == torch.float8_e4m3fn and keys.shape[1:] == (INDEX_HEAD_DIM,)
    assert weights.shape == (rows, heads)
    assert lens.shape == (rows,)
    if candidate_blocks is not None:
        assert candidate_blocks.ndim == 2 and candidate_blocks.shape[0] == rows
        assert candidate_blocks.dtype == torch.int32
        assert candidate_block_size in (1, 2, 4, 8, 16, 32, 64)
        assert candidate_blocks.stride(1) == 1
        width = candidate_blocks.shape[1] * candidate_block_size
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    keys = keys.contiguous()
    out_width = ((width + 3) // 4) * 4
    storage = torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    out = storage[:, :width]
    if rows == 0 or width == 0:
        return out
    block_l = 64
    _fp8_index_logits_prefill_kernel[(rows, triton.cdiv(out_width, block_l))](
        q,
        weights,
        keys,
        lens.to(torch.int64).contiguous(),
        candidate_blocks,
        out,
        key_width,
        out_width,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        H=heads,
        D=INDEX_HEAD_DIM,
        BLOCK_L=block_l,
        USE_CANDIDATES=candidate_blocks is not None,
        NUM_BLOCKS=candidate_blocks.shape[1] if candidate_blocks is not None else 0,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        num_warps=4,
    )
    return out
