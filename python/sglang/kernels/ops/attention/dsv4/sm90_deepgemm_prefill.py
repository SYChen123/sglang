"""Opt-in Hopper MQA scorer for the prefill indexer's gathered E4M3 operands.

This uses FP32 post-dot arithmetic. It does not reproduce the Triton/Torch
scorer's BF16 dot, product and head-sum rounding points.
"""

from __future__ import annotations

import torch


def deepgemm_fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    *,
    key_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score one request's contiguous K; each row sees ``keys[:lens[row]]``.

    ``q`` is [rows, 32|64, 128] E4M3, ``keys`` is [keys, 128] E4M3,
    ``weights`` is [rows, heads], and lengths are nonnegative and <= key count.
    The shared FP4-to-FP8 conversion already applies block scales, so runtime
    passes unit FP32 key scales. No MXFP4 or packed UE8M0 operand reaches DG.

    Each request has its own gathered K, hence all starts are zero. Use full
    width plus cleaning: candidate reduction needs -inf causal tails, whereas
    the compact-output API forbids ``clean_logits=True``. DG's padded backing
    stride also satisfies ragged top-k's vector-read alignment.
    """
    assert q.ndim == 3 and q.dtype == torch.float8_e4m3fn
    rows, heads, dim = q.shape
    assert heads in (32, 64) and dim == 128
    assert keys.ndim == 2 and keys.shape[1] == dim
    assert keys.dtype == torch.float8_e4m3fn
    assert weights.shape == (rows, heads) and lens.shape == (rows,)
    width = keys.shape[0]
    if rows == 0 or width == 0:
        return torch.empty((rows, width), device=q.device, dtype=torch.float32)
    import deep_gemm

    if not hasattr(deep_gemm, "fp8_mqa_logits"):
        raise RuntimeError("SM90 prefill needs DeepGEMM's fp8_mqa_logits API")
    if key_scales is None:
        key_scales = torch.ones(width, device=keys.device, dtype=torch.float32)
    assert key_scales.dtype == torch.float32 and key_scales.shape == (width,)
    ends = lens.to(torch.int32).contiguous()
    return deep_gemm.fp8_mqa_logits(
        q.contiguous(),
        (keys.contiguous(), key_scales.contiguous()),
        # Match the head-weight input of the BF16-rounding scorer, but DG's
        # SM90 kernel requires FP32 storage for those values.
        weights.to(torch.bfloat16).float().contiguous(),
        torch.zeros_like(ends),
        ends,
        True,
        0,
    )
