"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import (
    MoeA2ABackend,
    MoeRunnerBackend,
    mega_moe,
    mega_moe_sm90,
)
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_exec, get_flags, get_parallel

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDeepGemmMegaMoeApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        self.deep_gemm = ModuleType("deep_gemm")
        self.deep_gemm.num_sms = 132
        self.deep_gemm.get_num_sms = MagicMock(
            side_effect=lambda: self.deep_gemm.num_sms
        )
        self.deep_gemm.set_num_sms = MagicMock(
            side_effect=lambda num_sms: setattr(self.deep_gemm, "num_sms", num_sms)
        )
        max_num_sms = patch.object(mega_moe, "_mega_moe_max_num_sms", return_value=130)
        self.max_num_sms = max_num_sms.start()
        self.addCleanup(max_num_sms.stop)

    def tearDown(self):
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        super().tearDown()

    def test_mxf4_buffer_uses_typed_api(self):
        deep_gemm = self.deep_gemm
        expected_buffer = object()
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(return_value=expected_buffer)
        group = object()

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffer = mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="mxf4xmxf4",
            )

        self.assertIs(actual_buffer, expected_buffer)
        call = deep_gemm.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp8_dispatch", call.kwargs)

    def test_buffer_cache_separates_mma_types(self):
        deep_gemm = self.deep_gemm
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                    mma_type=mma_type,
                )
                for mma_type in ("fp8xfp4", "mxf4xmxf4")
            )

        self.assertEqual(actual_buffers, expected_buffers)
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["mma_type"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["fp8xfp4", "mxf4xmxf4"],
        )

    def test_buffer_cache_uses_effective_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                num_sms=deep_gemm.get_num_sms()
            )
        )
        group = object()
        buffers = []

        for current_num_sms, expected_num_sms in (
            (132, 130),
            (131, 130),
            (96, 96),
            (97, 96),
            (132, 130),
        ):
            with self.subTest(current_num_sms=current_num_sms):
                deep_gemm.set_num_sms(current_num_sms)
                buf = self._get_test_buffer(group)
                buffers.append(buf)
                self.assertEqual(buf.num_sms, expected_num_sms)
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)
                with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
                    self.assertEqual(buf.num_sms, deep_gemm.get_num_sms())
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)

        self.assertIs(buffers[0], buffers[1])
        self.assertIs(buffers[0], buffers[4])
        self.assertIs(buffers[2], buffers[3])
        self.assertIsNot(buffers[0], buffers[2])
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)

    def test_buffer_allocation_without_sm_override(self):
        deep_gemm = self.deep_gemm
        self.max_num_sms.return_value = None
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: deep_gemm.get_num_sms()
        )

        buf = self._get_test_buffer(object())
        self.assertEqual(buf, 132)
        deep_gemm.set_num_sms.assert_not_called()

    def test_buffer_allocation_failure_restores_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=RuntimeError("allocation failed")
        )

        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self._get_test_buffer(object())

        self.assertEqual(deep_gemm.get_num_sms(), 132)
        self.assertEqual(deep_gemm.set_num_sms.call_args_list, [call(130), call(132)])
        self.assertEqual(mega_moe._MEGA_MOE_SYMM_BUFFER, {})

    def test_mxf4_weight_transform_uses_matching_mma_type(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.transform_sf_into_required_layout = MagicMock(
            side_effect=lambda _sf, mn, k, recipe, num_groups, disable_ue8m0_cast: (
                torch.zeros((num_groups, mn, max(1, k // 32)), dtype=torch.int32)
            )
        )
        deep_gemm.transform_weights_for_mega_moe = MagicMock(
            side_effect=lambda l1, l2, **_kwargs: (l1, l2)
        )
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
        ):
            method.process_weights_after_loading(layer)

        call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")

    def test_sm90_fp4_loading_keeps_packed_storage_and_replaces_layout(self):
        """Do not retain a second expert layout or expand the experts to FP8."""
        for raw_scales in (False, True):
            with self.subTest(raw_ue8m0=raw_scales):
                names = (
                    ("w13_weight_scale", "w2_weight_scale")
                    if raw_scales
                    else ("w13_weight_scale_inv", "w2_weight_scale_inv")
                )
                experts = SimpleNamespace(
                    w13_weight=torch.nn.Parameter(
                        torch.zeros((1, 256, 64), dtype=torch.int8), requires_grad=False
                    ),
                    w2_weight=torch.nn.Parameter(
                        torch.zeros((1, 128, 64), dtype=torch.int8), requires_grad=False
                    ),
                )
                for name, n in zip(names, (256, 128)):
                    setattr(
                        experts,
                        name,
                        torch.nn.Parameter(
                            torch.full(
                                (1, n, 4),
                                127 if raw_scales else 1.0,
                                dtype=torch.uint8 if raw_scales else torch.float32,
                            ),
                            requires_grad=False,
                        ),
                    )
                transformed = (
                    (
                        torch.ones_like(experts.w13_weight),
                        torch.zeros((1, 256, 1), dtype=torch.int32),
                    ),
                    (
                        torch.ones_like(experts.w2_weight),
                        torch.zeros((1, 128, 1), dtype=torch.int32),
                    ),
                )
                dg = self.deep_gemm
                dg.fp8_fp4_mega_moe = MagicMock()
                dg.mega_moe_pre_dispatch_sm90 = MagicMock()
                dg.transform_weights_for_mega_moe_sm90_fp4 = MagicMock(
                    return_value=transformed
                )
                with patch.dict(sys.modules, {"deep_gemm": dg}):
                    mega_moe_sm90.build_sm90_fp4_mega_moe_experts_weights(
                        experts, scale_names=names
                    )
                    # Re-entering weight preparation must not interleave twice.
                    mega_moe_sm90.build_sm90_fp4_mega_moe_experts_weights(
                        experts, scale_names=names
                    )
                dg.transform_weights_for_mega_moe_sm90_fp4.assert_called_once()
                for param, scale_name, pair in zip(
                    (experts.w13_weight, experts.w2_weight), names, transformed
                ):
                    self.assertEqual(param.dtype, torch.int8)
                    self.assertEqual(param.data_ptr(), pair[0].data_ptr())
                    self.assertEqual(
                        getattr(experts, scale_name).data_ptr(), pair[1].data_ptr()
                    )
                    self.assertTrue(getattr(experts, scale_name).format_ue8m0)
                self.assertTrue(experts._mega_moe_sm90_fp4_weights)
                self.assertTrue(experts._mega_moe_weights_built)
                # Raw checkpoint exponents become exact powers of two for the
                # upstream transform, not the SM100 MN-major scale layout.
                for pair in dg.transform_weights_for_mega_moe_sm90_fp4.call_args.args:
                    self.assertEqual(pair[1].dtype, torch.float32)
                    self.assertTrue(torch.equal(pair[1], torch.ones_like(pair[1])))

    def test_sm90_fp4_tail_selects_weights_for_uniform_prefill_rows(self):
        dg = self.deep_gemm
        dg.mega_moe_pre_dispatch_sm90 = MagicMock()
        dg.fp8_fp4_mega_moe = MagicMock()
        dg.get_sm90_fp4_register_max_tokens = MagicMock(return_value=63)
        dg.pack_sm90_fp4_register_weights = MagicMock(
            side_effect=lambda w: w.unsqueeze(-1)
        )
        shapes = ((96, 4608, 2560), (96, 5120, 1152))
        weights = [torch.empty(s, dtype=torch.int8, device="meta") for s in shapes]
        scales = [torch.empty((e, n, k // 16), device="meta") for e, n, k in shapes]
        transformed = tuple(
            (w, torch.empty((e, n, k // 64), dtype=torch.int32, device="meta"))
            for w, (e, n, k) in zip(weights, shapes)
        )
        dg.transform_weights_for_mega_moe_sm90_fp4 = MagicMock(return_value=transformed)
        experts = SimpleNamespace(
            _mega_moe_sm90_fp4_prepare_tail_rs=True,
            w13_weight=torch.nn.Parameter(weights[0], requires_grad=False),
            w2_weight=torch.nn.Parameter(weights[1], requires_grad=False),
            w13_weight_scale_inv=torch.nn.Parameter(scales[0], requires_grad=False),
            w2_weight_scale_inv=torch.nn.Parameter(scales[1], requires_grad=False),
        )
        buf = SimpleNamespace(x=None, x_sf=None, topk_idx=None, topk_weights=None)
        with patch.dict(sys.modules, {"deep_gemm": dg}):
            mega_moe_sm90.build_sm90_fp4_mega_moe_experts_weights(experts)
            for prefill, rows, cp_rows, expected_rs in (
                (True, 0, [0] * 4, False),
                (True, 1, [1] * 4, True),
                (True, 32, [32] * 4, True),
                (True, 63, [63] * 4, True),
                (True, 64, [64] * 4, False),
                (True, 319, [319] * 4, False),
                (True, 320, [320] * 4, False),
                (True, 4096, [4096] * 4, False),
                (True, 32, [32, 32, 64, 64], False),
                (True, 32, None, False),
                (False, 32, [32] * 4, False),
            ):
                with self.subTest(prefill=prefill, rows=rows, cp_rows=cp_rows):
                    batch = SimpleNamespace(
                        forward_mode=SimpleNamespace(
                            is_extend_without_speculative=lambda: prefill,
                        ),
                        attn_cp_metadata=SimpleNamespace(per_rank_actual_token=cp_rows),
                    )
                    mega_moe_sm90.run_sm90_mega_routed(
                        experts,
                        torch.empty((rows, 16)),
                        torch.empty((rows, 6), dtype=torch.int32),
                        torch.empty((rows, 6)),
                        buf,
                        rows,
                        hidden_size=16,
                        forward_batch=batch,
                    )
                    call_args = dg.fp8_fp4_mega_moe.call_args
                    for actual, original in zip(call_args.args[1:3], transformed):
                        self.assertEqual(actual[0].ndim, 4 if expected_rs else 3)
                        self.assertTrue(torch._C._is_alias_of(actual[1], original[1]))
                    self.assertNotIn("disable_sm90_swap_ab", call_args.kwargs)
        self.assertEqual(dg.pack_sm90_fp4_register_weights.call_count, 2)

    def test_sm90_fp4_rejects_old_api_and_fp4_activations(self):
        dg = ModuleType("deep_gemm")
        dg.fp8_fp4_mega_moe = MagicMock()  # SM100 symbol alone is insufficient.
        experts = SimpleNamespace(_mega_moe_sm90_fp4_weights=True)
        with (
            patch.dict(sys.modules, {"deep_gemm": dg}),
            patch.object(mega_moe_sm90, "_device_sm", 90),
        ):
            self.assertFalse(mega_moe_sm90.is_sm90_fp4_mega_moe_available(experts))
            with self.assertRaisesRegex(RuntimeError, "DeepGEMM PR #53"):
                mega_moe_sm90.build_sm90_fp4_mega_moe_experts_weights(experts)
            experts._mega_moe_w4a4 = True
            with self.assertRaisesRegex(ValueError, "not FP4 activations"):
                mega_moe_sm90.build_sm90_fp4_mega_moe_experts_weights(experts)

    def test_sm90_fp4_local_rows_empty_rank_and_graph_buffer_reuse(self):
        dg = self.deep_gemm
        self.max_num_sms.return_value = None
        buf = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        dg.get_symm_buffer_for_mega_moe = MagicMock(return_value=buf)
        dg.mega_moe_pre_dispatch_sm90 = MagicMock()
        dg.transform_weights_for_mega_moe_sm90_fp4 = MagicMock()
        dg.fp8_fp4_mega_moe = MagicMock(side_effect=lambda y, *_a, **_k: y.fill_(7))
        dg.fp8_mega_moe = MagicMock(side_effect=lambda y, *_a, **_k: y.fill_(5))
        experts = SimpleNamespace(
            num_experts=8,
            _mega_moe_nvfp4=False,
            _mega_moe_w4a4=False,
            _mega_moe_weights_built=True,
            _mega_moe_sm90_fp4_weights=True,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
        )
        from sglang.srt.environ import envs

        with (
            patch.dict(sys.modules, {"deep_gemm": dg}),
            patch.object(mega_moe, "_device_sm", 90),
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=object())
            ),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(8),
        ):
            # Zero rows must still enter both distributed kernels. Physical
            # graph buckets are passed intact; logical pad ids remain -1.
            for rows in (0, 2, 8, 2):
                ids = torch.full((rows, 2), -1, dtype=torch.int64)
                hidden = torch.zeros((rows * 2, 128), dtype=torch.bfloat16)[::2]
                out = mega_moe.run_mega_routed_experts(
                    experts,
                    hidden,
                    ids,
                    torch.zeros((rows, 2), dtype=torch.bfloat16),
                    hidden_size=128,
                    intermediate_size=128,
                    top_k=2,
                    num_tokens=rows,
                    activation_clamp=10.0,
                    routed_scaling_factor=1.5,
                )
                self.assertEqual(out.shape, (rows, 128))
                self.assertTrue(torch.equal(out, torch.full_like(out, 7)))
                pre = dg.mega_moe_pre_dispatch_sm90.call_args
                self.assertEqual(pre.kwargs["num_tokens"], rows)
                self.assertEqual(pre.kwargs["group_size"], 128)
                self.assertEqual(pre.kwargs["routed_scaling_factor"], 1.5)
                self.assertTrue(pre.args[0].is_contiguous())
                self.assertTrue(torch.equal(pre.args[0], hidden))
                self.assertTrue(torch.equal(pre.args[1], ids.to(torch.int32)))
                self.assertEqual(
                    dg.fp8_fp4_mega_moe.call_args.kwargs["recipe"], (1, 1, 32)
                )
                self.assertEqual(
                    dg.fp8_fp4_mega_moe.call_args.args[0].shape[0], max(rows, 1)
                )
            dg.get_symm_buffer_for_mega_moe.assert_called_once()
            experts._mega_moe_sm90_fp4_weights = False
            out = mega_moe.run_mega_routed_experts(
                experts,
                torch.zeros((2, 128), dtype=torch.bfloat16),
                torch.zeros((2, 2), dtype=torch.int64),
                torch.ones((2, 2)),
                hidden_size=128,
                intermediate_size=128,
                top_k=2,
                num_tokens=2,
            )
            self.assertTrue(torch.equal(out, torch.full_like(out, 5)))
            self.assertEqual(
                dg.fp8_mega_moe.call_args.kwargs["recipe"], (128, 128, 128)
            )

    def test_sm90_fp4_cp_local_budget_and_capture_overflow(self):
        """A CP shard fits independently; a larger captured bucket cannot fall back."""
        from sglang.srt.environ import envs

        dg = self.deep_gemm
        dg.fp8_fp4_mega_moe = MagicMock()
        dg.mega_moe_pre_dispatch_sm90 = MagicMock()
        dg.transform_weights_for_mega_moe_sm90_fp4 = MagicMock()
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                _mega_moe_weights_built=True, _mega_moe_sm90_fp4_weights=True
            )
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": dg}),
            patch.object(mega_moe, "_device_sm", 90),
            patch.object(mega_moe_sm90, "_device_sm", 90),
            patch.object(mega_moe, "get_dp_global_num_tokens", return_value=[32] * 4),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(8),
        ):
            with (
                patch.object(mega_moe, "get_is_capture_mode", return_value=False),
                patch.object(mega_moe, "is_dsa_enable_prefill_cp", return_value=True),
            ):
                self.assertTrue(
                    mega_moe.should_use_mega_moe(moe, torch.empty((8, 128)))
                )
            with patch.object(mega_moe, "get_is_capture_mode", return_value=True):
                self.assertTrue(
                    mega_moe.should_use_mega_moe(moe, torch.empty((8, 128)))
                )
                with self.assertRaisesRegex(
                    RuntimeError, "fallback would use repacked"
                ):
                    mega_moe.should_use_mega_moe(moe, torch.empty((9, 128)))

    def test_mxf4_pre_dispatch_uses_typed_api(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=True,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                swiglu_limit=None,
            ),
            experts=experts,
            gate=MagicMock(
                return_value=torch.empty((1, 8)), e_score_correction_bias_vl=None
            ),
            topk=MagicMock(return_value=topk_output),
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=0,
            routed_scaling_factor=1.0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(
                mega_moe,
                "_get_mega_moe_symm_buffer",
                return_value=buffer,
            ),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=object())
            ),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((1, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=1,
            )

        self.assertTrue(deep_gemm.mega_moe_pre_dispatch.called)
        call = deep_gemm.mega_moe_pre_dispatch.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp4_acts", call.kwargs)

    def test_run_mega_routed_experts_generic_entry(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        hidden_states = torch.zeros((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(mega_moe, "mega_moe_pre_dispatch") as pre_dispatch,
            patch.object(
                mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
            ) as get_buffer,
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
            ),
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                hidden_states,
                topk_ids,
                topk_weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                activation_clamp=7.0,
                routed_scaling_factor=1.0,
            )

        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, torch.bfloat16)
        buf_call = get_buffer.call_args
        self.assertEqual(buf_call.kwargs.get("num_topk"), 2)
        self.assertEqual(buf_call.kwargs.get("hidden"), 4)
        self.assertEqual(buf_call.kwargs.get("intermediate_hidden"), 8)
        # The kernel wants int32 ids and fp32 weights regardless of the router dtype.
        ids_arg, weights_arg = pre_dispatch.call_args.args[1:3]
        self.assertEqual(ids_arg.dtype, torch.int32)
        self.assertEqual(weights_arg.dtype, torch.float32)
        mega_call = deep_gemm.fp8_fp4_mega_moe.call_args
        self.assertIs(mega_call.args[1], experts.mega_l1_weights)
        self.assertIs(mega_call.args[2], experts.mega_l2_weights)
        self.assertEqual(mega_call.kwargs.get("activation_clamp"), 7.0)

    def test_v41_megamoe_selects_image_bias_on_local_rows(self):
        """Image rows must retain their router bias after attention-TP sharding.

        Equal logits make the two biases select disjoint experts. A numerical
        expert boundary exposes accidentally routing image rows as text, and
        padded rows must contribute nothing.
        """
        for fused_shared in (0, 1):
            for scale_in_topk in (False, True):
                with self.subTest(
                    fused_shared=fused_shared, scale_in_topk=scale_in_topk
                ):
                    moe = self._v41_moe(fused_shared, scale_in_topk)
                    ids = torch.tensor([7, 99, 99])
                    hidden = torch.ones((3, 4))
                    batch = SimpleNamespace(
                        moe_num_token_non_padded=lambda: torch.tensor(
                            2, dtype=torch.int32
                        )
                    )

                    def run_experts(_experts, x, topk_ids, weights, **kwargs):
                        # Routed experts multiply x by their physical ID + 1;
                        # the rank-1 shared slot (ID 5) is an identity expert.
                        factors = (topk_ids + 1).float()
                        factors[topk_ids == 5] = 1.0
                        self.assertTrue((topk_ids[2] == -1).all())
                        return x * (
                            (factors * weights).sum(-1, keepdim=True)
                            * kwargs["routed_scaling_factor"]
                        )

                    with (
                        get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
                        get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
                        patch.object(
                            mega_moe.ExpertLocationDispatchInfo,
                            "init_new",
                            return_value=None,
                        ),
                        patch.object(mega_moe, "run_mega_routed_experts", run_experts),
                        # The fused padded-row fill requires Triton/GPU.
                        patch(
                            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                            return_value=False,
                        ),
                        patch.object(topk_module, "_is_cuda", False),
                        patch.object(
                            topk_module, "_can_fuse_padded_region", return_value=False
                        ),
                    ):
                        out = mega_moe._run_mega_routed(
                            moe, hidden, batch, ids, num_tokens=3
                        )
                        text_out = mega_moe._run_mega_routed(
                            moe, hidden, batch, None, num_tokens=3
                        )
                    # Text selects logical 0/1; image selects logical 2/3. With
                    # a shared slot, rank-1 routed IDs shift by one (2/3 -> 3/4).
                    expected = torch.tensor(
                        [3.0 + fused_shared, 7.0 + 3.0 * fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(out, expected)
                    text_expected = torch.tensor(
                        [3.0 + fused_shared, 3.0 + fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(text_out, text_expected)

    def test_v41_megamoe_empty_rank_still_participates(self):
        """An idle DP rank must reach A2A combine without running its router."""
        moe = self._v41_moe(0, False)
        hidden = torch.empty((0, 4))

        def run_experts(_experts, x, ids, weights, **kwargs):
            self.assertIsNone(ids)
            self.assertIsNone(weights)
            # The collective boundary owns the result even on an idle rank.
            return x.new_empty((0, kwargs["hidden_size"]))

        with patch.object(mega_moe, "run_mega_routed_experts", run_experts):
            out = mega_moe._run_mega_routed(
                moe, hidden, None, torch.empty(0, dtype=torch.long), num_tokens=0
            )
        self.assertEqual(out.shape, (0, 4))
        moe.gate.assert_not_called()

    @staticmethod
    def _v41_moe(fused_shared, scale_in_topk):
        return SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                image_token_id=99,
            ),
            experts=SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=scale_in_topk,
                moe_runner_config=SimpleNamespace(swiglu_limit=None),
            ),
            gate=MagicMock(
                return_value=torch.zeros((3, 4)),
                e_score_correction_bias=torch.tensor([4.0, 3.0, 0.0, 0.0]),
                e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 3.0, 4.0]),
            ),
            topk=SimpleNamespace(
                topk_config=TopKConfig(
                    top_k=2 + fused_shared,
                    num_fused_shared_experts=fused_shared,
                    renormalize=True,
                    routed_scaling_factor=2.0,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                )
            ),
            is_hash=False,
            num_fused_shared_experts=fused_shared,
            layer_id=0,
            routed_scaling_factor=2.0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
        )

    def test_shape_check_uses_pinned_fp8_fp4_alignment(self):
        # Packed SF groups need 128 elements, not a 16-byte token SF row.
        mega_moe.check_mega_moe_shapes(2048, 768, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(5120, 2304, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(512, 128, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 128"):
            mega_moe.check_mega_moe_shapes(2048, 736, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(2016, 768, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(256, 128, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1536, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1024, "mxf4xmxf4")
        # 768 is a multiple of 256, so the NVFP4 (g16) rule accepts it.
        mega_moe.check_mega_moe_shapes(2048, 768, "nvfp4xnvfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 256"):
            mega_moe.check_mega_moe_shapes(2048, 384, "nvfp4xnvfp4")

    def test_mxf4_l1_uses_packed_gate_up_interleave(self):
        source = torch.arange(32).reshape(1, 32)
        expected = torch.tensor(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
            ]
        ).reshape(1, 32)

        actual = mega_moe._interleave_mega_moe_gate_up(source, gran=16)

        torch.testing.assert_close(actual, expected)

    def test_draft_layers_keep_their_own_w4a4_choice(self):
        """Draft MegaMoE layers must not follow the target's MMA type once built.

        The draft forward runs outside draft_model_build_scope, so a layer that
        re-read the global flag would pair draft weights with the wrong kernel.
        """
        for target_flag, draft_flag, expected_target, expected_draft in (
            (True, False, "mxf4xmxf4", "fp8xfp4"),
            (True, None, "mxf4xmxf4", "mxf4xmxf4"),
            (False, True, "fp8xfp4", "mxf4xmxf4"),
            (False, None, "fp8xfp4", "fp8xfp4"),
        ):
            with self.subTest(
                enable_w4a4_mxfp4_megamoe=target_flag,
                speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
            ):
                with get_context().override_server_args(
                    model_path="dummy",
                    enable_w4a4_mxfp4_megamoe=target_flag,
                    speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
                ):
                    target = self._build_fused_moe()
                    with draft_model_build_scope():
                        draft = self._build_fused_moe()

                    self.assertEqual(
                        get_exec().moe.enable_w4a4_mxfp4_megamoe, target_flag
                    )
                    self.assertEqual(
                        mega_moe._mega_moe_mma_type(target), expected_target
                    )
                    self.assertEqual(mega_moe._mega_moe_mma_type(draft), expected_draft)

    def test_draft_build_scope_restores_w4a4_on_exception(self):
        """A failed draft build must not leave the target on the draft's MMA type."""
        with get_context().override_server_args(
            model_path="dummy",
            enable_w4a4_mxfp4_megamoe=False,
            speculative_enable_w4a4_mxfp4_megamoe=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "draft build failed"):
                with draft_model_build_scope():
                    self.assertTrue(get_exec().moe.enable_w4a4_mxfp4_megamoe)
                    raise RuntimeError("draft build failed")

            self.assertFalse(get_exec().moe.enable_w4a4_mxfp4_megamoe)
            target = self._build_fused_moe()
            self.assertEqual(mega_moe._mega_moe_mma_type(target), "fp8xfp4")

    def _build_fused_moe(self):
        method = UnquantizedFusedMoEMethod()
        with (
            patch.object(method, "create_weights"),
            patch.object(method, "create_moe_runner"),
            patch.object(
                fused_moe_layer_module,
                "create_moe_dispatcher",
                return_value=SimpleNamespace(),
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.AUTO,
                a2a_backend=MoeA2ABackend.MEGAMOE,
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            return fused_moe_layer_module.FusedMoE(
                num_experts=2,
                hidden_size=4,
                intermediate_size=8,
                layer_id=0,
                quant_method=method,
            )

    def _get_test_buffer(self, group):
        with patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}):
            return mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="fp8xfp4",
            )


if __name__ == "__main__":
    unittest.main()
