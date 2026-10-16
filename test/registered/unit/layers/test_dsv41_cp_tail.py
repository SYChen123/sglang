"""Exercise compact bounded-replay output gathering using simulated CP ranks."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend"
MODEL = "sglang.srt.models.deepseek_v4"
INTERLEAVE = "sglang.srt.layers.cp.interleave"
CP = "sglang.srt.layers.cp.utils"


class TestDSV41CPTail(CustomTestCase):
    def test_tail_moe_gather_preserves_integer_ids_and_rank_major_padding(self):
        for dtype in (torch.int32, torch.int64):
            for cp_size in (2, 8):
                for use_a2a in (False, True):
                    for rank in range(cp_size):
                        with self.subTest(
                            dtype=dtype, cp_size=cp_size, rank=rank, a2a=use_a2a
                        ):
                            self._check_tail_moe_ids(dtype, cp_size, rank, use_a2a)

    def _check_tail_moe_ids(self, dtype, cp_size, rank, use_a2a):
        # Values must not round through BF16/FP16, or truncate int64 to int32.
        ids = torch.arange(20, dtype=dtype) + 129250
        if dtype == torch.int64:
            ids += 1 << 40
        indices = torch.tensor([0, 1, 16, 17, 18, 19])
        lengths = torch.tensor([2, 3, 1])
        batch = SimpleNamespace(
            batch_size=3,
            input_ids=ids,
            positions=torch.arange(20),
            forward_mode=ForwardMode.EXTEND,
        )
        parallel = SimpleNamespace(
            attn_cp_rank=rank,
            attn_cp_size=cp_size,
            attn_cp_group=object(),
            attn_tp_size=1,
            attn_dp_size=1,
        )
        backend = object.__new__(DeepseekV4AttnBackend)
        with patch(f"{BACKEND}.get_parallel", return_value=parallel):
            layout = backend._late_layer_tail_cp_layout(batch, indices, lengths)
        physical = layout["cp_metadata"].per_rank_actual_token[rank]
        chunks = []
        for r in range(cp_size):
            chunk = ids[indices[indices % cp_size == r]]
            chunks.append(torch.cat([chunk, chunk.new_zeros(physical - chunk.numel())]))
        expected_ids = torch.cat(chunks)
        tail = LateLayerTail(
            token_indices=layout["local_token_indices"],
            positions=layout["local_positions"],
            extend_seq_lens=lengths,
            extend_seq_lens_cpu=lengths.tolist(),
            swa_out_cache_loc=torch.empty(0),
            pad_rows=layout["pad_rows"],
            cp_metadata=layout["cp_metadata"],
            local_lens_cpu=layout["local_lens_cpu"],
        )
        backend.tail_forward_metadata = SimpleNamespace(late_layer_tail=tail)
        backend.enter_late_layer_tail = Mock(return_value=object())
        backend.exit_late_layer_tail = Mock()
        id_outputs = []

        def gather(out, local):
            # Model the real collective's dtype/shape checks, which copy_ alone
            # would hide by silently casting IDs into the hidden-state buffer.
            if out.dtype != local.dtype:
                raise TypeError("output tensor must have the same type as input tensor")
            self.assertEqual(out.numel(), cp_size * local.numel())
            self.assertEqual(out.shape[1:], local.shape[1:])
            self.assertTrue(local.is_contiguous())
            if local.is_floating_point():
                out.fill_(42)
            else:
                torch.testing.assert_close(local.squeeze(-1), chunks[rank])
                out.copy_(expected_ids.reshape_as(out))
                id_outputs.append(out)

        padded = torch.cat([ids, ids.new_zeros(-ids.numel() % cp_size)])
        full_global_ids = padded.view(-1, cp_size).T.flatten()

        def layer_forward(layer, **kwargs):
            local, global_ids = kwargs["input_ids"], kwargs["input_ids_global"]
            if layer == 0:
                torch.testing.assert_close(global_ids, full_global_ids)
            else:
                torch.testing.assert_close(local.squeeze(-1), chunks[rank])
                expected = chunks[rank] if use_a2a else expected_ids
                torch.testing.assert_close(global_ids, expected)
                if not use_a2a:
                    from sglang.srt.layers.cp.interleave import (
                        attn_cp_interleave_gather,
                    )

                    attn_cp_interleave_gather(
                        torch.zeros(physical, 3, dtype=torch.bfloat16)
                    )
                    torch.testing.assert_close(global_ids, expected)
            return kwargs["state"]

        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=1),
            engram_hasher=None,
            engram_prefetch_stream=None,
            late_layer_start=1,
            start_layer=0,
            end_layer=3,
            _check_late_layer_tail_readers=Mock(),
            layers=[
                SimpleNamespace(
                    engram=None,
                    hc_cfg=None,
                    forward_hc_pre_from_prev=lambda i=i, **kw: layer_forward(i, **kw),
                )
                for i in range(3)
            ],
        )
        with (
            patch(f"{MODEL}.is_cp_active", return_value=True),
            patch(f"{MODEL}.get_parallel", return_value=parallel),
            patch(
                f"{MODEL}.get_forward", return_value=SimpleNamespace(sp_active=False)
            ),
            patch(f"{MODEL}.get_attn_backend", return_value=backend),
            patch(
                f"{MODEL}.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: not use_a2a),
            ),
            patch(f"{MODEL}.check_cuda_graph_backend", return_value=True),
            patch(f"{MODEL}.use_symmetric_memory", return_value=nullcontext()),
            patch(f"{MODEL}.is_allocation_symmetric", return_value=False),
            patch(f"{INTERLEAVE}.get_parallel", return_value=parallel),
            patch(f"{INTERLEAVE}.use_symmetric_memory", return_value=nullcontext()),
            patch(f"{INTERLEAVE}.is_allocation_symmetric", return_value=False),
            patch(f"{INTERLEAVE}.attn_cp_all_gather_into_tensor", side_effect=gather),
        ):
            local_ids = padded[rank::cp_size]
            out = DeepseekV4Model._forward_layers_hc_pre_from_prev(
                model,
                torch.zeros(local_ids.numel()),
                torch.zeros(local_ids.numel(), 2, 3, dtype=torch.bfloat16),
                batch,
                local_ids,
                full_global_ids,
                False,
                [],
            )
        self.assertIs(out[2], tail)
        self.assertEqual(len(id_outputs), 0 if use_a2a else 1)
        backend.enter_late_layer_tail.assert_called_once_with(batch)
        backend.exit_late_layer_tail.assert_called_once_with(
            backend.enter_late_layer_tail.return_value, batch
        )

    def test_tail_gather_preserves_dspark_global_indices_and_empty_ranks(self):
        cp_size = 8
        token_indices = torch.tensor([0, 1, 16, 17, 18, 19])
        lengths = torch.tensor([2, 3, 1])
        # These are global logical tail rows. The collective below packs them
        # rank-major; the real gather_index must restore request order.
        global_outputs = [
            torch.stack(
                [token_indices.float() + k, token_indices.float() + k + 0.5], -1
            )
            for k in range(4)
        ]
        global_outputs.append(token_indices)
        for rank in range(cp_size):
            with self.subTest(rank=rank):
                full_cp = object()
                batch = SimpleNamespace(
                    batch_size=3,
                    forward_mode=ForwardMode.EXTEND,
                    positions=torch.arange(20),
                    input_ids=torch.arange(20) + 100,
                    attn_cp_metadata=full_cp,
                )
                parallel = SimpleNamespace(
                    attn_cp_rank=rank, attn_cp_size=cp_size, attn_cp_group=None
                )
                backend = object.__new__(DeepseekV4AttnBackend)
                with patch(f"{BACKEND}.get_parallel", return_value=parallel):
                    layout = backend._late_layer_tail_cp_layout(
                        batch, token_indices, lengths
                    )
                meta = layout["cp_metadata"]
                tail = LateLayerTail(
                    token_indices=layout["local_token_indices"],
                    positions=layout["local_positions"],
                    extend_seq_lens=lengths,
                    extend_seq_lens_cpu=lengths.tolist(),
                    swa_out_cache_loc=torch.empty(0),
                    pad_rows=layout["pad_rows"],
                    cp_metadata=meta,
                    local_lens_cpu=layout["local_lens_cpu"],
                )
                strategy = InterleaveCPStrategy(cp_size=cp_size)
                owned = token_indices % cp_size == rank
                physical = meta.per_rank_actual_token[rank]

                def local_tensor(value, owned=owned, physical=physical):
                    local = value[owned]
                    pad = local.new_zeros((physical - local.shape[0], *local.shape[1:]))
                    return torch.cat([local, pad])

                collective_calls = []

                def gather(
                    out,
                    local,
                    collective_calls=collective_calls,
                    physical=physical,
                    local_tensor=local_tensor,
                ):
                    value = global_outputs[len(collective_calls)]
                    torch.testing.assert_close(local, local_tensor(value))
                    chunks = []
                    for r in range(cp_size):
                        chunk = value[token_indices % cp_size == r]
                        chunks.append(
                            torch.cat(
                                [
                                    chunk,
                                    chunk.new_zeros(
                                        (physical - len(chunk), *chunk.shape[1:])
                                    ),
                                ]
                            )
                        )
                    out.copy_(torch.cat(chunks))
                    collective_calls.append(local)

                logits = Mock(return_value=SimpleNamespace())
                model = SimpleNamespace(
                    capture_aux_hidden_states=True,
                    model=SimpleNamespace(late_layer_start=21),
                    logits_processor=logits,
                    lm_head=object(),
                )
                with (
                    patch(
                        f"{MODEL}.get_attn_backend",
                        return_value=SimpleNamespace(
                            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
                        ),
                    ),
                    patch(f"{MODEL}.get_parallel", return_value=parallel),
                    patch(
                        f"{MODEL}.LogitsMetadata",
                        SimpleNamespace(from_forward_batch=lambda _: SimpleNamespace()),
                    ),
                    patch("torch.cuda.current_stream", return_value=None),
                    patch(f"{CP}.is_cp_active", return_value=True),
                    patch(f"{CP}.get_cp_strategy", return_value=strategy),
                    # cp_rank is inherited from the base strategy and reads
                    # its module's runtime context, not an instance attribute.
                    patch(
                        "sglang.srt.layers.cp.base.get_parallel", return_value=parallel
                    ),
                    patch(f"{INTERLEAVE}.get_parallel", return_value=parallel),
                    patch(
                        f"{INTERLEAVE}.use_symmetric_memory", return_value=nullcontext()
                    ),
                    patch(f"{INTERLEAVE}.is_allocation_symmetric", return_value=False),
                    patch(
                        f"{INTERLEAVE}.attn_cp_all_gather_into_tensor",
                        side_effect=gather,
                    ),
                ):
                    self.assertEqual(strategy.cp_rank, rank)
                    out = DeepseekV4ForCausalLM.logits_from_cp_hidden_states(
                        model,
                        (
                            local_tensor(global_outputs[0]),
                            local_tensor(global_outputs[1]),
                        ),
                        [
                            local_tensor(global_outputs[2]),
                            local_tensor(global_outputs[3]),
                        ],
                        batch,
                    )
                self.assertIs(batch.attn_cp_metadata, full_cp)
                self.assertEqual(len(collective_calls), 5)
                torch.testing.assert_close(
                    out.hidden_states_token_indices, token_indices
                )
                args = logits.call_args.args
                torch.testing.assert_close(args[0], batch.input_ids[token_indices])
                torch.testing.assert_close(args[1], global_outputs[0])
                torch.testing.assert_close(args[4][0], global_outputs[2])
                self.assertEqual(args[3].extend_seq_lens_cpu, [2, 3, 1])
                self.assertIsNone(logits.call_args.kwargs["hidden_states_before_norm"])

    def test_cp_layout_restored_when_gather_raises(self):
        full_cp, tail_cp = object(), object()
        batch = SimpleNamespace(
            attn_cp_metadata=full_cp, forward_mode=ForwardMode.EXTEND
        )
        model = SimpleNamespace(
            capture_aux_hidden_states=True, model=SimpleNamespace(late_layer_start=21)
        )
        tail = SimpleNamespace(cp_metadata=tail_cp)
        with (
            patch(
                f"{MODEL}.get_attn_backend",
                return_value=SimpleNamespace(
                    tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
                ),
            ),
            patch("torch.cuda.current_stream", return_value=None),
            patch(
                f"{CP}.cp_gather_after_forward",
                side_effect=RuntimeError("collective failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "collective failed"),
        ):
            DeepseekV4ForCausalLM.logits_from_cp_hidden_states(model, None, None, batch)
        self.assertIs(batch.attn_cp_metadata, full_cp)


if __name__ == "__main__":
    unittest.main()
