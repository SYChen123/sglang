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
from sglang.srt.layers.attention.dsv4.cp_bcg import CPTailGraphBuffers
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


class TestDSV41CPTailGraph(CustomTestCase):
    def _batch(self, cp_size):
        return SimpleNamespace(
            input_ids=torch.arange(64, dtype=torch.int64) + 1000,
            positions=torch.arange(64, dtype=torch.int64),
            out_cache_loc=torch.arange(64, dtype=torch.int64) + 256,
            req_pool_indices=torch.tensor([2, 3, 4]),
            batch_size=3,
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=SimpleNamespace(
                per_rank_actual_token=[64 // cp_size] * cp_size
            ),
        )

    def _layout(self, batch, indices, lengths, cp_size, rank):
        parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_size=cp_size)
        with patch(f"{BACKEND}.get_parallel", return_value=parallel):
            return DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                None, batch, indices, lengths, local_capacity=8
            )

    def test_changed_layout_reuses_addresses_and_masks_every_padding_row(self):
        cases = [
            (torch.tensor([0, 1, 16, 17, 18, 19]), torch.tensor([2, 3, 1])),
            (torch.tensor([8, 9, 10, 11]), torch.tensor([1, 2, 1])),
        ]
        for cp_size in (2, 4, 8):
            for rank in range(cp_size):
                batch = self._batch(cp_size)
                buffers = CPTailGraphBuffers.create(
                    batch, local_rows=8, cp_size=cp_size
                )
                pointers = {
                    name: value.data_ptr() for name, value in vars(buffers).items()
                }
                for indices, lengths in cases:
                    with self.subTest(
                        cp_size=cp_size, rank=rank, indices=indices.tolist()
                    ):
                        layout = self._layout(batch, indices, lengths, cp_size, rank)
                        meta = buffers.update(
                            batch, indices, lengths, layout, cp_rank=rank
                        )
                        self.assertEqual(
                            pointers,
                            {
                                name: value.data_ptr()
                                for name, value in vars(buffers).items()
                            },
                        )
                        self.assertEqual(meta.per_rank_actual_token, [8] * cp_size)
                        self.assertEqual(meta.per_rank_logical_token, [8] * cp_size)
                        self.assertEqual(meta.total_seq_lens, 8 * cp_size)
                        torch.testing.assert_close(
                            meta.gather_index.sort().values, torch.arange(8 * cp_size)
                        )
                        packed = torch.full((cp_size, 8), -999, dtype=torch.int64)
                        for r in range(cp_size):
                            owned = indices[indices % cp_size == r]
                            packed[r, : owned.numel()] = owned
                        torch.testing.assert_close(
                            packed.flatten()[meta.gather_index][: indices.numel()],
                            indices,
                        )
                        real = int(buffers.num_token_non_padded)
                        self.assertEqual(real, int((indices % cp_size == rank).sum()))
                        torch.testing.assert_close(
                            buffers.out_cache_loc[: indices.numel()],
                            batch.out_cache_loc[indices],
                        )
                        self.assertTrue(
                            (buffers.out_cache_loc[indices.numel() :] == 0).all()
                        )
                        self.assertTrue(
                            (buffers.scatter_indices[real:] == 64 // cp_size).all()
                        )
                        # Scattering graph padding must never overwrite a real
                        # token, including row zero on ranks with no tail rows.
                        local_ids = batch.input_ids[rank::cp_size]
                        tail_ids = local_ids[buffers.token_indices]
                        scattered = torch.zeros(
                            local_ids.numel() + 1, dtype=local_ids.dtype
                        )
                        scattered.index_copy_(0, buffers.scatter_indices, tail_ids)
                        valid_indices = layout["local_token_indices"]
                        torch.testing.assert_close(
                            scattered[valid_indices], local_ids[valid_indices]
                        )
                        if real == 0:
                            self.assertTrue((scattered[:-1] == 0).all())

    def test_graph_metadata_refresh_preserves_cp_tail_storage_and_window(self):
        for local_rows in (None, 68):
            with self.subTest(local_rows=local_rows):
                self._check_graph_metadata_refresh(local_rows)

    def _check_graph_metadata_refresh(self, local_rows):
        for rank in range(4):
            parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_size=4)
            backend = object.__new__(DeepseekV4AttnBackend)
            backend.encoder_replay = False
            backend._bcg_tail_capacity = 4 * 128
            backend._bcg_tail_metadata = {}
            backend.req_to_token = torch.arange(4 * 1024).view(4, 1024).to(torch.int32)
            backend.page_size, backend.index_topk = 256, 512
            backend.present_ratios, backend.low_ratios = (0,), ()
            backend.has_c4 = backend.has_c128 = backend.trtllm_attn = False
            backend.cuda_int32_kwargs = {"dtype": torch.int32, "device": "cpu"}
            backend.token_to_kv_pool = SimpleNamespace(
                request_window=SimpleNamespace(capacity=256)
            )
            held = None
            pointers = None
            for ext, prefix in [
                ([41, 159, 85], [0, 384, 0]),
                ([20, 133], [0, 256]),
                ([145], [500]),
            ]:
                lengths = torch.tensor(ext)
                positions = torch.cat(
                    [torch.arange(p, p + n) for p, n in zip(prefix, ext)]
                )
                floors = torch.repeat_interleave(torch.tensor(prefix), lengths)
                backend.encoder_row_floor = floors
                ids = torch.arange(288, dtype=torch.int64)
                real_locs = torch.cat(
                    [
                        backend.req_to_token[r, p : p + n].long()
                        for r, p, n in zip([1, 2, 3], prefix, ext)
                    ]
                )
                batch = SimpleNamespace(
                    input_ids=ids,
                    prefill_decoder_tail_local_rows=local_rows,
                    positions=torch.nn.functional.pad(
                        positions, (0, 288 - len(positions))
                    ),
                    out_cache_loc=torch.nn.functional.pad(
                        real_locs, (0, 288 - len(real_locs))
                    ),
                    req_pool_indices=torch.arange(1, len(ext) + 1),
                    batch_size=len(ext),
                    extend_seq_lens_cpu=ext,
                    seq_lens_cpu=torch.tensor([p + n for p, n in zip(prefix, ext)]),
                    seq_lens=torch.tensor([p + n for p, n in zip(prefix, ext)]),
                    forward_mode=ForwardMode.EXTEND,
                    attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[72] * 4),
                )
                with (
                    patch(f"{BACKEND}.get_parallel", return_value=parallel),
                    patch(f"{BACKEND}.is_cp_active", return_value=True),
                    patch(f"{BACKEND}._create_flashmla_metadata", return_value=None),
                ):
                    metadata = backend._prefill_graph_tail_metadata(batch, 1024)
                if held is not None:
                    self.assertIs(metadata, held)
                    self.assertEqual(
                        pointers,
                        (
                            metadata.late_layer_tail.token_indices.data_ptr(),
                            metadata.late_layer_tail.positions.data_ptr(),
                            metadata.core_metadata.raw_out_loc.data_ptr(),
                            metadata.late_layer_tail.cp_metadata.gather_index.data_ptr(),
                        ),
                    )
                held = metadata
                tail = metadata.late_layer_tail
                pointers = (
                    tail.token_indices.data_ptr(),
                    tail.positions.data_ptr(),
                    metadata.core_metadata.raw_out_loc.data_ptr(),
                    tail.cp_metadata.gather_index.data_ptr(),
                )
                physical = 72 if local_rows is None else local_rows
                self.assertEqual(
                    metadata.core_metadata.raw_out_loc.shape, (physical * 4,)
                )
                self.assertEqual(
                    metadata.core_metadata.positions_casual.shape, (physical,)
                )
                self.assertEqual(tail.positions.shape, (physical,))
                self.assertEqual(tail.extend_seq_lens_cpu, [min(128, n) for n in ext])
                real_global = sum(tail.extend_seq_lens_cpu)
                self.assertTrue(
                    (metadata.core_metadata.raw_out_loc[real_global:] == 0).all()
                )
                self.assertFalse(
                    metadata.core_metadata.request_window_layout.commit_mask[
                        real_global:
                    ].any()
                )
                self.assertTrue(
                    metadata.core_metadata.request_window_layout.commit_mask[
                        :real_global
                    ].all()
                )
                self.assertEqual(
                    int(
                        metadata.core_metadata.request_window_layout.history_pos.reshape(
                            4, 128
                        )[len(ext) - 1, -1]
                    ),
                    prefix[-1] + max(0, ext[-1] - 128) - 1,
                )
                self.assertEqual(
                    int(tail.cp_metadata.moe_local_token_count),
                    sum(tail.local_lens_cpu),
                )

    def test_graph_tail_logits_drop_padding_and_keep_dspark_token_indices(self):
        cp_size = 8
        indices, lengths = torch.tensor([0, 1, 16, 17, 18, 19]), torch.tensor([2, 3, 1])
        for rank in range(cp_size):
            with self.subTest(rank=rank):
                batch = self._batch(cp_size)
                original_cp = batch.attn_cp_metadata
                layout = self._layout(batch, indices, lengths, cp_size, rank)
                buffers = CPTailGraphBuffers.create(
                    batch, local_rows=8, cp_size=cp_size
                )
                meta = buffers.update(batch, indices, lengths, layout, cp_rank=rank)
                tail = LateLayerTail(
                    token_indices=buffers.token_indices,
                    positions=buffers.positions,
                    extend_seq_lens=lengths,
                    extend_seq_lens_cpu=lengths.tolist(),
                    swa_out_cache_loc=buffers.swa_out_cache_loc,
                    cp_metadata=meta,
                    scatter_indices=buffers.scatter_indices,
                    real_rows_count=int(buffers.num_token_non_padded),
                )
                logits = Mock(return_value=SimpleNamespace())
                model = SimpleNamespace(
                    capture_aux_hidden_states=True,
                    model=SimpleNamespace(late_layer_start=21),
                    logits_processor=logits,
                    lm_head=object(),
                )
                outputs = [indices[:, None].float() + i for i in range(4)] + [indices]
                calls = []

                def gather(value, forward_batch, stream):
                    self.assertIs(forward_batch.attn_cp_metadata, meta)
                    n = len(calls)
                    calls.append(value)
                    if isinstance(value, tuple):
                        calls.append(value[1])
                        return tuple(
                            torch.cat(
                                [outputs[i], torch.zeros(8 * cp_size - len(indices), 1)]
                            )
                            for i in (0, 1)
                        )
                    result = outputs[n]
                    return torch.cat(
                        [
                            result,
                            result.new_zeros(
                                (8 * cp_size - len(indices), *result.shape[1:])
                            ),
                        ]
                    )

                local_hidden = torch.ones(8, 1)
                with (
                    patch(
                        f"{MODEL}.get_attn_backend",
                        return_value=SimpleNamespace(
                            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
                        ),
                    ),
                    patch(
                        f"{MODEL}.get_parallel",
                        return_value=SimpleNamespace(
                            attn_cp_rank=rank, attn_cp_size=cp_size
                        ),
                    ),
                    patch(
                        f"{MODEL}.LogitsMetadata",
                        SimpleNamespace(from_forward_batch=lambda _: SimpleNamespace()),
                    ),
                    patch("torch.cuda.current_stream", return_value=None),
                    patch(f"{CP}.cp_gather_after_forward", side_effect=gather),
                ):
                    output = DeepseekV4ForCausalLM.logits_from_cp_hidden_states(
                        model,
                        (local_hidden, local_hidden),
                        [local_hidden, local_hidden],
                        batch,
                    )
                self.assertIs(batch.attn_cp_metadata, original_cp)
                torch.testing.assert_close(output.hidden_states_token_indices, indices)
                args = logits.call_args.args
                torch.testing.assert_close(args[0], batch.input_ids[indices])
                torch.testing.assert_close(args[1], outputs[0])
                torch.testing.assert_close(args[4][0], outputs[2])
                torch.testing.assert_close(args[4][1], outputs[3])


class TestDSV41CPTailBuckets(CustomTestCase):
    def test_three_requests_use_96_local_rows_in_the_8192_encoder_bucket(self):
        """BS3 must not inherit the max64 request tail (2048 local rows)."""
        backend = object.__new__(DeepseekV4AttnBackend)
        backend._bcg_tail_capacity = 8192
        batch = SimpleNamespace(
            prefill_decoder_tail_local_rows=96,
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[2048] * 4),
        )
        with (
            patch(f"{BACKEND}.is_cp_active", return_value=True),
            patch(
                f"{BACKEND}.get_parallel", return_value=SimpleNamespace(attn_cp_size=4)
            ),
        ):
            self.assertEqual(backend._prefill_graph_tail_block_rows(batch), 384)

    def test_cpu_dispatch_covers_short_tails_with_the_same_cp_owner(self):
        from sglang.srt.layers.attention.dsv4.cp_bcg import (
            cp_tail_local_buckets,
            cp_tail_local_rows,
        )

        for cp_size in (2, 4, 8):
            # The first case has multiple short tails with the same owner;
            # ceil(aggregate / CP) alone underestimates that rank's rows.
            for lengths in ([1, 131, 1], [41, 159, 85], [4096] * 3, [1]):
                indices, offset = [], 0
                for n in lengths:
                    indices.extend(range(offset + n - min(128, n), offset + n))
                    offset += n
                expected = max(
                    sum(i % cp_size == rank for i in indices) for rank in range(cp_size)
                )
                self.assertEqual(
                    cp_tail_local_rows(lengths, window=128, cp_size=cp_size), expected
                )
                capacity = 64 * ((128 + cp_size - 1) // cp_size)
                buckets = cp_tail_local_buckets(capacity, window=128, cp_size=cp_size)
                selected = next(n for n in buckets if n >= expected)
                self.assertGreaterEqual(selected, expected)
                if lengths == [4096] * 3:
                    self.assertEqual(selected, 384 // cp_size)

    def test_attention_reads_only_live_rows_and_zeros_static_bridge_padding(self):
        for tail_rows in ([32, 32, 32], None, []):
            with self.subTest(tail_rows=tail_rows):
                live = 96 if tail_rows else (3 if tail_rows is None else 0)
                query = torch.arange(128 * 2).reshape(128, 1, 2).float()
                output = torch.full_like(query, float("nan"))
                tail = (
                    None
                    if tail_rows is None
                    else SimpleNamespace(local_lens_cpu=tail_rows)
                )
                batch = SimpleNamespace(
                    extend_seq_lens_cpu=[11], out_cache_loc=torch.arange(16)
                )

                def attention(**kwargs):
                    self.assertEqual(kwargs["q"].shape[0], live)
                    self.assertEqual(kwargs["k"].shape[0], live)
                    return kwargs["q"] + 5

                backend = SimpleNamespace(
                    forward_metadata=SimpleNamespace(late_layer_tail=tail),
                    forward=attention,
                )
                with (
                    patch(f"{MODEL}.is_cp_active", return_value=True),
                    patch(
                        f"{MODEL}.get_parallel",
                        return_value=SimpleNamespace(attn_cp_rank=0, attn_cp_size=4),
                    ),
                    patch(f"{MODEL}.get_attn_backend", return_value=backend),
                    patch(
                        f"{MODEL}.get_tc_piecewise_forward_context",
                        return_value=SimpleNamespace(
                            forward_batch=batch, attention_layers=[object()]
                        ),
                    ),
                ):
                    # Exercise the actual registered Python implementation;
                    # its tensor math runs on CPU with a stub attention backend.
                    torch.ops.sglang.deepseek_v4_attention_with_output.default.redispatch(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA),
                        query,
                        query,
                        output,
                        0,
                        0,
                        torch.zeros(1),
                        False,
                    )
                torch.testing.assert_close(output[:live], query[:live] + 5)
                self.assertTrue((output[live:] == 0).all())

    def test_empty_query_rank_still_commits_the_global_window(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        commits = []
        backend.head_dim_v = 2
        backend.trtllm_attn = False
        backend.is_dspark_draft = False
        backend.token_to_kv_pool = SimpleNamespace(
            request_window=SimpleNamespace(commit=commits.append),
            _swa_local_layer_id=lambda layer: layer,
        )
        backend._forward_attention = Mock(
            side_effect=AssertionError("empty attention kernel")
        )
        q = torch.empty(0, 1, 2)
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with patch(f"{BACKEND}.is_cp_active", return_value=True):
            output = backend.forward(
                q, q, q, SimpleNamespace(layer_id=21), batch, save_kv_cache=False
            )
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(commits, [21])


if __name__ == "__main__":
    unittest.main()
