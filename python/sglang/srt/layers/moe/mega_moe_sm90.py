# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SM90 W8A8 / packed-W4A8 MegaMoE forward and expert-weight preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.models.deepseek_common.utils import _device_sm

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer


def is_sm90_fp8_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return (
        hasattr(deep_gemm, "fp8_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and getattr(experts, "_mega_moe_sm90_fp8_weights", False)
    )


def is_sm90_fp4_mega_moe_available(experts) -> bool:
    if _device_sm != 90 or not getattr(experts, "_mega_moe_sm90_fp4_weights", False):
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    # The generic fp8_fp4_mega_moe symbol also exists in SM100-only builds.
    return all(
        hasattr(deep_gemm, name)
        for name in (
            "fp8_fp4_mega_moe",
            "mega_moe_pre_dispatch_sm90",
            "transform_weights_for_mega_moe_sm90_fp4",
        )
    )


def run_sm90_mega_routed(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
    *,
    hidden_size: int,
    activation_clamp: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    import deep_gemm

    deep_gemm.mega_moe_pre_dispatch_sm90(
        hidden_states.contiguous(),
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=128,
        routed_scaling_factor=routed_scaling_factor,
    )

    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    use_fp4 = getattr(experts, "_mega_moe_sm90_fp4_weights", False)
    mega_moe = deep_gemm.fp8_fp4_mega_moe if use_fp4 else deep_gemm.fp8_mega_moe
    mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32) if use_fp4 else (128, 128, 128),
        activation="swiglu",
        activation_clamp=activation_clamp,
        fast_math=True,
    )
    y = y[:num_tokens]

    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)


def build_sm90_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    assert w13.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    num_groups, n1, k1 = w13.shape
    _, n2, k2 = w2.shape
    scale_group_mn, scale_group_k = 128, 128

    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid SM90 mega-moe K/group_size: k1={k1}, k2={k2}, group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    w13_interleaved = _interleave_l1_weight_only(w13)
    experts.w13_weight.data = w13_interleaved
    experts.mega_l1_weights = (
        experts.w13_weight.data,
        experts.w13_weight_scale_inv.data,
    )
    experts.mega_l2_weights = (
        experts.w2_weight.data,
        experts.w2_weight_scale_inv.data,
    )

    experts._mega_moe_sm90_fp8_weights = True
    experts._mega_moe_weights_built = True


def build_sm90_fp4_mega_moe_experts_weights(
    experts,
    *,
    scale_names: tuple[str, str] = ("w13_weight_scale_inv", "w2_weight_scale_inv"),
) -> None:
    """Keep packed E2M1 weights; pack K32 scales for the Hopper fused kernel.

    The DeepGEMM PR #53 transform interleaves gate/up in gran-8 rows and packs
    four UE8M0 bytes per int32 along K. This is not SM100's MN-major SF layout.
    Replace parameters before KV sizing so a checkpoint-layout copy is not kept.
    """
    if getattr(experts, "_mega_moe_weights_built", False):
        return
    if getattr(experts, "_mega_moe_w4a4", False):
        raise ValueError(
            "SM90 MegaMoE supports W4A8, not FP4 activations; disable "
            "--enable-w4a4-mxfp4-megamoe and its speculative counterpart."
        )
    config = getattr(experts, "moe_runner_config", None)
    if config is not None and (
        config.activation not in ("silu", "swiglu")
        or not config.is_gated
        or config.gemm1_alpha is not None
        or config.apply_router_weight_on_input
        or getattr(experts, "with_bias", False)
    ):
        raise ValueError(
            "The SM90 W4A8 MegaMoE adapter requires bias-free SwiGLU experts "
            "with routing weights applied after the down projection."
        )
    import deep_gemm

    if not all(
        hasattr(deep_gemm, name)
        for name in (
            "fp8_fp4_mega_moe",
            "mega_moe_pre_dispatch_sm90",
            "transform_weights_for_mega_moe_sm90_fp4",
        )
    ):
        raise RuntimeError(
            "SM90 W4A8 MegaMoE requires a build of DeepGEMM PR #53 "
            "(b5675bf7c812a583d6eadabfa23ad1e00207c39d). The released "
            "SM100-only fp8_fp4_mega_moe API is insufficient; see "
            "3rdparty/DeepGEMM and scripts/sm90_w4a8_megamoe/README.md."
        )

    w13, w2 = experts.w13_weight.data, experts.w2_weight.data
    scales = [getattr(experts, name) for name in scale_names]
    for weight, scale in zip((w13, w2), scales):
        if weight.dtype not in (torch.int8, torch.uint8):
            raise ValueError("SM90 W4A8 MegaMoE requires packed E2M1 expert weights")
        e, n, packed_k = weight.shape
        if packed_k % 64 or scale.shape != (e, n, packed_k // 16):
            raise ValueError(
                "SM90 W4A8 MegaMoE requires K aligned to 128 and per-row K32 "
                f"scales; got weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
            )
    if w13.shape[0] != w2.shape[0] or w13.shape[1] != w2.shape[2] * 4:
        raise ValueError("SM90 W4A8 MegaMoE requires matching gate/up and down experts")
    if w13.shape[2] * 2 != w2.shape[1]:
        raise ValueError("SM90 W4A8 MegaMoE hidden dimensions do not match")
    if not 0 < w2.shape[2] * 2 <= 4096:
        raise ValueError("SM90 W4A8 MegaMoE requires 0 < intermediate_size <= 4096")

    fp32_scales = []
    for scale in scales:
        if scale.dtype == torch.uint8:
            fp32_scales.append(scale.data.view(torch.float8_e8m0fnu).float())
        elif scale.dtype == torch.float32:
            fp32_scales.append(scale.data)
        else:
            raise ValueError(
                f"Expected FP32 or raw UE8M0 expert scales, got {scale.dtype}"
            )

    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
        (w13.view(torch.int8), fp32_scales[0]),
        (w2.view(torch.int8), fp32_scales[1]),
    )
    experts.w13_weight.data, scales[0].data = l1
    experts.w2_weight.data, scales[1].data = l2
    for scale in scales:
        scale.format_ue8m0 = True
    experts.mega_l1_weights = (experts.w13_weight.data, scales[0].data)
    experts.mega_l2_weights = (experts.w2_weight.data, scales[1].data)
    experts._mega_moe_sm90_fp4_weights = True
    experts._mega_moe_weights_built = True
