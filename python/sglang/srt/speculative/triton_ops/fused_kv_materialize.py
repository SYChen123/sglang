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
"""Fused Triton kernel for DFlash KV materialization.

Combines: KV projection (cuBLAS) + RMSNorm + RoPE (Triton), then pool-managed KV writes.
Set SGLANG_DFLASH_NORM_ROPE_TILELANG=1 to use the TileLang backend instead.
"""

import os
from typing import Callable, List

import torch
import triton
import triton.language as tl

# pip install apache-tvm-ffi==0.1.9
# pip install tilelang==0.1.8
_USE_TILELANG = os.environ.get("SGLANG_DFLASH_NORM_ROPE_TILELANG", "0") == "1"


# @cached_triton_kernel(
#     lambda args, kwargs: (args[0].dtype, args[1].dtype, args[14], args[15], args[16], args[18])
#     # key = (kv_dtype, norm_weight_dtype, head_dim, kv_size, rotary_dim, eps)
#     # - kv_dtype covers k_out/v_out (empty_like)
#     # - positions is always int64, cos_sin_cache dtype is fixed per model
#     # - half_rotary_dim = rotary_dim // 2, BLOCK_HD = next_power_of_2(head_dim): both derived
# )
@triton.jit
def _fused_norm_rope_kernel(
    kv_ptr,  # [total_ctx, kv_size * 2]
    k_norm_weight_ptr,  # [head_dim]
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [total_ctx]
    k_out_ptr,  # [total_ctx, num_kv_heads, head_dim]
    v_out_ptr,  # [total_ctx, num_kv_heads, head_dim]
    kv_stride_ctx: tl.constexpr,
    cos_sin_stride_pos: tl.constexpr,
    k_out_stride_ctx: tl.constexpr,
    k_out_stride_head: tl.constexpr,
    v_out_stride_ctx: tl.constexpr,
    v_out_stride_head: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load metadata
    position = tl.load(positions_ptr + ctx_id)

    # Compute base pointers
    kv_base = kv_ptr + ctx_id * kv_stride_ctx
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    k_write = k_out_ptr + ctx_id * k_out_stride_ctx + head_id * k_out_stride_head
    v_write = v_out_ptr + ctx_id * v_out_stride_ctx + head_id * v_out_stride_head

    # Load K and V
    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    # RMSNorm on K
    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(k_norm_weight_ptr + offs, mask=mask_hd, other=1.0).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    # RoPE (neox style): k_first, k_second -> rotated
    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    # Extract first/second halves of K for rotation
    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr + half_rotary_dim + offs, mask=mask_half, other=1.0
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    # Apply rotation
    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    # Store V (no transform)
    tl.store(v_write + offs, v_raw, mask=mask_hd)

    # Store K: rotated halves + pass-through
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


# ---------------------------------------------------------------------------
# TileLang backend
# ---------------------------------------------------------------------------


def _make_fused_norm_rope_tilelang(
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    _tl_dtype: str,
    _tl_norm_dtype: str,
    eps: float,
):
    """Return a compiled TileLang kernel for the given shape/dtype config.

    Uses T.symbolic() for total_ctx and max_pos with @T.prim_func (lazy mode)
    and NO out_idx – outputs are passed as pre-allocated parameters by the
    caller.  This avoids the ~200us per-launch overhead of out_idx which
    internally allocates output tensors on every call.

    Follows the act_quant_kernel pattern from the NSA tilelang kernels.
    """
    import tilelang
    import tilelang.language as T

    half_rotary_dim = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)
    kv_size = num_kv_heads * head_dim

    # Symbolic dimensions – resolved at runtime from input shapes,
    # no recompilation needed for different total_ctx values.
    total_ctx_sym = T.symbolic("total_ctx")
    max_pos_sym = T.symbolic("max_pos")

    @tilelang.jit(target="cuda")
    def _kernel_factory():
        @T.prim_func
        def main(
            kv: T.Tensor[(total_ctx_sym, kv_size * 2), _tl_dtype],
            k_norm_weight: T.Tensor[(head_dim,), _tl_norm_dtype],
            cos_sin_cache: T.Tensor[(max_pos_sym, rotary_dim), "float32"],
            positions: T.Tensor[(total_ctx_sym,), "int64"],
            k_out: T.Tensor((total_ctx_sym, num_kv_heads, head_dim), _tl_dtype),
            v_out: T.Tensor((total_ctx_sym, num_kv_heads, head_dim), _tl_dtype),
        ):
            # Grid: (total_ctx, num_kv_heads), BLOCK_HD threads per block
            with T.Kernel(total_ctx_sym, num_kv_heads, threads=BLOCK_HD) as (
                ctx_id,
                head_id,
            ):
                # ── load K, V ────────────────────────────────────────────────
                k_frag = T.alloc_fragment([BLOCK_HD], "float32")
                v_frag = T.alloc_fragment([BLOCK_HD], _tl_dtype)
                norm_w = T.alloc_fragment([BLOCK_HD], "float32")

                for i in T.Parallel(BLOCK_HD):
                    j = head_id * head_dim + i
                    if i < head_dim:
                        k_frag[i] = T.cast(kv[ctx_id, j], "float32")
                        v_frag[i] = kv[ctx_id, kv_size + j]
                        norm_w[i] = T.cast(k_norm_weight[i], "float32")
                    else:
                        k_frag[i] = T.float32(0.0)

                # ── RMSNorm ──────────────────────────────────────────────────
                k_sq = T.alloc_fragment([BLOCK_HD], "float32")
                for i in T.Parallel(BLOCK_HD):
                    k_sq[i] = k_frag[i] * k_frag[i]  # 0.0 for i >= head_dim

                sq_sum = T.alloc_fragment([1], "float32")
                T.reduce_sum(k_sq, sq_sum, dim=0, clear=True)

                inv_rms = T.rsqrt(sq_sum[0] / T.float32(head_dim) + T.float32(eps))

                k_normed = T.alloc_fragment([BLOCK_HD], "float32")
                for i in T.Parallel(BLOCK_HD):
                    k_normed[i] = k_frag[i] * inv_rms * norm_w[i]

                # ── load cos/sin and second-half K ────────────────────────────
                position = positions[ctx_id]
                cos_frag = T.alloc_fragment([BLOCK_HD], "float32")
                sin_frag = T.alloc_fragment([BLOCK_HD], "float32")
                k_second = T.alloc_fragment([BLOCK_HD], "float32")

                for i in T.Parallel(BLOCK_HD):
                    if i < half_rotary_dim:
                        cos_frag[i] = cos_sin_cache[position, i]
                        sin_frag[i] = cos_sin_cache[position, half_rotary_dim + i]
                        j2 = head_id * head_dim + half_rotary_dim + i
                        k_second[i] = (
                            T.cast(kv[ctx_id, j2], "float32")
                            * inv_rms
                            * T.cast(k_norm_weight[half_rotary_dim + i], "float32")
                        )
                    else:
                        cos_frag[i] = T.float32(1.0)
                        sin_frag[i] = T.float32(0.0)
                        k_second[i] = T.float32(0.0)

                # ── RoPE + write K ────────────────────────────────────────────
                for i in T.Parallel(BLOCK_HD):
                    if i < half_rotary_dim:
                        k_out[ctx_id, head_id, i] = T.cast(
                            k_normed[i] * cos_frag[i] - k_second[i] * sin_frag[i],
                            _tl_dtype,
                        )
                        k_out[ctx_id, head_id, half_rotary_dim + i] = T.cast(
                            k_second[i] * cos_frag[i] + k_normed[i] * sin_frag[i],
                            _tl_dtype,
                        )
                    else:
                        # pass-through region [rotary_dim, head_dim)
                        if rotary_dim <= i and i < head_dim:
                            k_out[ctx_id, head_id, i] = T.cast(k_normed[i], _tl_dtype)

                # ── write V (no transform) ────────────────────────────────────
                for i in T.Parallel(BLOCK_HD):
                    if i < head_dim:
                        v_out[ctx_id, head_id, i] = v_frag[i]

        return main

    return _kernel_factory()


# Cache: (dtype, norm_dtype, num_kv_heads, head_dim, rotary_dim, eps) → compiled kernel
# NOTE: total_ctx is NOT in the key – it is a T.symbolic() dimension,
#       so one compiled kernel serves all batch sizes.
_tilelang_kernel_cache: dict = {}


def _get_tilelang_kernel(num_kv_heads, head_dim, rotary_dim, dtype, norm_dtype, eps):
    key = (dtype, norm_dtype, num_kv_heads, head_dim, rotary_dim, eps)
    if key not in _tilelang_kernel_cache:
        _tl_dtype = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[dtype]
        _tl_norm_dtype = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[norm_dtype]
        _tilelang_kernel_cache[key] = _make_fused_norm_rope_tilelang(
            num_kv_heads, head_dim, rotary_dim, _tl_dtype, _tl_norm_dtype, eps
        )
    return _tilelang_kernel_cache[key]


def _fused_norm_rope(
    kv: torch.Tensor,  # [total_ctx, kv_size*2]
    k_norm_weight: torch.Tensor,  # [head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [total_ctx]
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + RoPE materialization for a single layer."""
    total_ctx = kv.shape[0]
    if total_ctx == 0:
        empty = torch.empty(
            (0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        return empty, empty

    kv_size = num_kv_heads * head_dim
    if kv.shape[1] != kv_size * 2:
        raise ValueError(
            "Invalid fused KV projection shape: "
            f"got {tuple(kv.shape)}, expected second dim {kv_size * 2}."
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            "Invalid fused KV rotary/head dim pair: "
            f"rotary_dim={rotary_dim}, head_dim={head_dim}."
        )

    # Ensure int64 for indexing
    if positions.device != kv.device:
        positions = positions.to(device=kv.device, dtype=torch.int64)
    elif positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    if _USE_TILELANG:
        kernel = _get_tilelang_kernel(
            num_kv_heads,
            head_dim,
            rotary_dim,
            kv.dtype,
            k_norm_weight.dtype,
            eps,
        )
        # Pre-allocate outputs and pass as params (no out_idx)
        k_out = torch.empty(
            (total_ctx, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        v_out = torch.empty_like(k_out)
        kernel(kv, k_norm_weight, cos_sin_cache, positions, k_out, v_out)
    else:
        k_out = torch.empty(
            (total_ctx, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        v_out = torch.empty_like(k_out)
        half_rotary_dim = rotary_dim // 2
        BLOCK_HD = triton.next_power_of_2(head_dim)
        _fused_norm_rope_kernel[(total_ctx, num_kv_heads)](
            kv,
            k_norm_weight,
            cos_sin_cache,
            positions,
            k_out,
            v_out,
            kv.stride(0),
            cos_sin_cache.stride(0),
            k_out.stride(0),
            k_out.stride(1),
            v_out.stride(0),
            v_out.stride(1),
            num_kv_heads,
            head_dim,
            kv_size,
            rotary_dim,
            half_rotary_dim,
            eps,
            BLOCK_HD,
        )

    return k_out, v_out


class FusedKVMaterializeHelper:
    """Fused KV materialization helper using batched projection.

    Uses torch.einsum for batched KV projection across all layers,
    then a Triton kernel for fused RMSNorm + RoPE materialization per layer.
    """

    def __init__(
        self,
        layers: List,
        rotary_emb,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.n_layers = len(layers)
        self.device = device

        self.rotary_dim = int(getattr(rotary_emb, "rotary_dim", head_dim))
        self.is_neox_style = bool(getattr(rotary_emb, "is_neox_style", True))

        if not self.is_neox_style:
            raise NotImplementedError("Only neox-style RoPE is supported.")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError(
                "Invalid fused KV rotary/head dim pair: "
                f"rotary_dim={self.rotary_dim}, head_dim={self.head_dim}."
            )

        # Pre-extract and stack weights for batched projection.
        kv_weights = []
        self.k_norm_weights = []
        self.eps_values = []

        for layer_id, layer in enumerate(layers):
            attn = layer.self_attn
            if int(attn.num_kv_heads) != self.num_kv_heads:
                raise ValueError(
                    "num_kv_heads mismatch across layers for fused KV path: "
                    f"expected {self.num_kv_heads}, got {int(attn.num_kv_heads)} at layer {layer_id}."
                )
            if int(attn.head_dim) != self.head_dim:
                raise ValueError(
                    "head_dim mismatch across layers for fused KV path: "
                    f"expected {self.head_dim}, got {int(attn.head_dim)} at layer {layer_id}."
                )
            layer_rotary_dim = int(
                getattr(attn.rotary_emb, "rotary_dim", self.head_dim)
            )
            layer_is_neox = bool(getattr(attn.rotary_emb, "is_neox_style", True))
            if (
                layer_rotary_dim != self.rotary_dim
                or layer_is_neox != self.is_neox_style
            ):
                raise ValueError(
                    "RoPE config mismatch across layers for fused KV path: "
                    f"expected (rotary_dim={self.rotary_dim}, neox={self.is_neox_style}), "
                    f"got (rotary_dim={layer_rotary_dim}, neox={layer_is_neox}) at layer {layer_id}."
                )

            # Extract KV portion of QKV weight
            qkv_w = attn.qkv_proj.weight
            kv_weight = qkv_w[attn.q_size : attn.q_size + 2 * attn.kv_size]
            kv_weights.append(kv_weight)
            self.k_norm_weights.append(attn.k_norm.weight)
            self.eps_values.append(attn.k_norm.variance_epsilon)

        # Stack for batched einsum: [n_layers, kv_size*2, hidden_size]
        self.batched_kv_weight = torch.stack(kv_weights)

    def materialize(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],
    ) -> None:
        """Materialize KV cache for all layers using batched projection."""
        total_ctx = ctx_hidden.shape[0]
        if total_ctx == 0:
            return

        if positions.ndim != 1:
            positions = positions.reshape(-1)
        if positions.numel() != total_ctx:
            raise ValueError(
                "positions must match ctx_hidden token count for fused KV materialization: "
                f"positions={positions.numel()}, total_ctx={total_ctx}."
            )

        max_position = int(positions.max().item())
        ensure_cos_sin_cache_length = getattr(
            self.rotary_emb, "_ensure_cos_sin_cache_length", None
        )
        if callable(ensure_cos_sin_cache_length):
            ensure_cos_sin_cache_length(max_position)

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        if max_position >= int(cos_sin_cache.shape[0]):
            raise RuntimeError(
                "RoPE cos/sin cache is too short for fused KV materialization: "
                f"max_position={max_position}, cache_len={int(cos_sin_cache.shape[0])}."
            )
        if cos_sin_cache.device != ctx_hidden.device:
            cos_sin_cache = cos_sin_cache.to(ctx_hidden.device)

        # Batched KV projection: [n_layers, total_ctx, kv_size*2]
        kv_all = torch.einsum("th,loh->lto", ctx_hidden, self.batched_kv_weight)
        if _USE_TILELANG and not kv_all.is_contiguous():
            # TileLang kernels assume C-contiguous layout (no stride params).
            # Ensure kv is contiguous – it may be a slice of the einsum output
            # kv_all[layer_id] which can have non-standard strides.
            kv_all = kv_all.contiguous()

        # Per-layer fused norm/RoPE/materialize, then delegate writes to the KV pool.
        for layer_id in range(self.n_layers):
            cache_k, cache_v = _fused_norm_rope(
                kv_all[layer_id],
                self.k_norm_weights[layer_id],
                cos_sin_cache,
                positions,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_dim,
                self.eps_values[layer_id],
            )
            write_layer_kv(layer_id, cache_k, cache_v)
