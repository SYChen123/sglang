# Copyright 2023-2026 SGLang Team
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

"""Padding helpers shared by context-parallel strategies."""

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner


def should_unpad_cp_decode_attn_tp(
    model_runner: Optional["ModelRunner"] = None,
) -> bool:
    """Decode/verify/draft rows are replicated under CP decode attention TP.

    Only interleave CP with a validated attention-TP architecture can skip CP
    alignment. Real context-parallel prefills still pad collective inputs.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.cp.cp_decode_attn_tp import (
        CP_DECODE_ATTN_TP_SUPPORTED_ARCHS,
    )

    model_supported = True
    if model_runner is not None:
        hf_config = getattr(model_runner.model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", None) or ()
        model_supported = any(
            arch in CP_DECODE_ATTN_TP_SUPPORTED_ARCHS for arch in architectures
        )
    parallel = get_parallel()
    return bool(
        getattr(parallel, "enable_cp_decode_attn_tp", False)
        and parallel.enable_prefill_cp
        and parallel.cp_strategy == "interleave"
        and parallel.dp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.attn_cp_size > 1
        and parallel.attn_tp_size == 1
        and model_supported
        and not envs.SGLANG_CP_DISABLE_DECODE_UNPADDING.get()
    )


def get_cp_padding_align_size(
    forward_mode: Optional["ForwardMode"] = None,
    model_runner: Optional["ModelRunner"] = None,
) -> int:
    """Return the token-count alignment required by the active CP strategy."""
    from sglang.srt.layers.cp.base import is_zigzag

    if (
        forward_mode is not None
        and not forward_mode.is_context_parallel_extend()
        and should_unpad_cp_decode_attn_tp(model_runner)
    ):
        return 1

    attn_cp_size = get_parallel().attn_cp_size
    if is_zigzag():
        return attn_cp_size * 2
    return attn_cp_size


def pad_logical_token_to_physical(metadata: Any) -> None:
    """Align each CP rank's physical token count for CP collectives."""
    logical_tokens = list(metadata.per_rank_actual_token)
    align_size = get_cp_padding_align_size()
    physical_rank_len = (
        (max(logical_tokens) + align_size - 1) // align_size * align_size
    )
    metadata.per_rank_logical_token = logical_tokens
    metadata.per_rank_actual_token = [physical_rank_len] * len(logical_tokens)
    metadata.max_rank_len = [physical_rank_len] * len(logical_tokens)


def pad_local_rows(x: torch.Tensor, metadata: Any, dim: int) -> torch.Tensor:
    """Pad a local CP tensor from its logical length to its physical length."""
    if (
        metadata.per_rank_logical_token is None
        or metadata.per_rank_logical_token == metadata.per_rank_actual_token
    ):
        return x

    target_len = metadata.per_rank_actual_token[0]
    pad_size = target_len - x.shape[dim]
    assert pad_size >= 0
    if pad_size == 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
