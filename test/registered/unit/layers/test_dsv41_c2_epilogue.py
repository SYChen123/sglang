"""C2 epilogue eligibility, dispatch and the pre-RoPE index-K contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4 import c2_epilogue_sm90 as kernels
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestC2Epilogue(CustomTestCase):
    def run_write(
        self, *, enabled=True, compatible=True, ratio=2, rope=64, owns_k=True
    ):
        backend = object.__new__(backend_module.DeepseekV4AttnBackend)
        pooled = torch.randn(6, 512)
        latent = torch.randn(6, 512, dtype=torch.bfloat16)
        old_latent = torch.zeros_like(latent)
        freqs = torch.ones(32, 32, dtype=torch.complex64)
        position = torch.arange(6)
        gathered = freqs[position]
        slots = torch.tensor([0, 128, 0, 129, 0, 130])
        index_keys = torch.randn(6, 128)
        layer = SimpleNamespace(
            layer_id=2,
            compress_ratio=ratio,
            rope_head_dim=rope,
            freqs_cis=freqs,
            compressor=SimpleNamespace(
                finish=Mock(return_value=old_latent),
                norm=SimpleNamespace(weight=torch.ones(512), eps=1e-20),
            ),
            indexer=SimpleNamespace(
                owns_k=owns_k, index_keys=Mock(return_value=index_keys)
            ),
        )
        pool = Mock()
        pool.uniform_fp8 = False
        pool.get_extra_key_layout.return_value = KVLayout.V4
        backend.token_to_kv_pool = pool
        with (
            patch.object(kernels, "can_use_c2_epilogue", return_value=compatible),
            patch.object(
                kernels, "c2_epilogue_sm90", return_value=(latent, gathered)
            ) as fused,
            patch.object(
                backend_module, "_rope_fq4", return_value=old_latent
            ) as old_rope,
        ):
            backend._low_ratio_write_group(
                layer, pooled, slots, position, fuse_main_store=enabled
            )
        use_fused = enabled and compatible and ratio == 2 and rope == 64
        self.assertEqual(fused.call_count, int(use_fused))
        self.assertEqual(layer.compressor.finish.call_count, int(not use_fused))
        self.assertEqual(old_rope.call_count, int(not use_fused))
        self.assertEqual(pool.set_extra_key_buffer_fused.call_count, int(not use_fused))
        if use_fused:
            self.assertIs(fused.call_args.args[0], pooled)
            self.assertIs(fused.call_args.args[4], slots)
            self.assertIs(fused.call_args.args[5], position)
        if owns_k:
            self.assertIs(
                layer.indexer.index_keys.call_args.args[0],
                latent if use_fused else old_latent,
            )
            torch.testing.assert_close(
                layer.indexer.index_keys.call_args.args[1], gathered
            )
            pool.set_index_k_fp4.assert_called_once_with(
                layer_id=2, loc=slots, cache_k=index_keys
            )
        else:
            layer.indexer.index_keys.assert_not_called()
            pool.set_index_k_fp4.assert_not_called()

    def test_pre_rope_latent_reaches_indexer_without_duplicate_main_store(self):
        self.run_write()
        self.run_write(owns_k=False)

    def test_opt_out_and_incompatible_layout_keep_old_path(self):
        for kwargs in (
            {"enabled": False},
            {"compatible": False},
            {"ratio": 1},
            {"rope": 32},
        ):
            with self.subTest(kwargs=kwargs):
                self.run_write(**kwargs)

    def test_only_hopper_enables_epilogue_after_pair_pooling(self):
        backend = object.__new__(backend_module.DeepseekV4AttnBackend)
        backend._low_ratio_write_group = Mock()
        backend._low_ratio_compress_fused = Mock()
        backend.token_to_kv_pool = Mock()
        pool = backend.token_to_kv_pool
        pool.uniform_fp8 = False
        pool.get_extra_key_layout.return_value = KVLayout.V4
        backend.token_to_kv_pool.get_attention_compress_states.return_value = (
            SimpleNamespace(
                ring_size=8,
                kv_score_buffer=SimpleNamespace(
                    shape=(17, 1024), kv=Mock(), score=Mock()
                ),
            )
        )
        backend.forward_metadata = SimpleNamespace(core_metadata=Mock())
        layer = SimpleNamespace(layer_id=2, compress_ratio=2, compressor=Mock())
        layer.compressor.use_fused_compress = False
        layer.compressor.project.return_value = (Mock(), Mock())
        x = SimpleNamespace(is_cuda=True)
        result = (Mock(), Mock(), Mock())
        with (
            patch(
                "sglang.kernels.ops.attention.dsv4.c2_decode_pool.c2_decode_pool",
                return_value=result,
            ),
            patch.object(torch.version, "cuda", "12.8"),
            patch.object(backend_module, "is_sm100_or_newer", return_value=False),
        ):
            for enabled in (False, True):
                for sm90 in (False, True):
                    with (
                        self.subTest(enabled=enabled, sm90=sm90),
                        patch.object(
                            backend_module.envs.SGLANG_OPT_DSV41_C2_EPILOGUE,
                            "get",
                            return_value=enabled,
                        ),
                        patch.object(
                            backend_module,
                            "get_platform",
                            return_value=SimpleNamespace(is_sm90=sm90),
                        ),
                    ):
                        backend._low_ratio_compress_decode(layer, x, Mock(), Mock())
                        self.assertEqual(
                            backend._low_ratio_write_group.call_args.kwargs[
                                "fuse_main_store"
                            ],
                            enabled and sm90,
                        )
            # The fused Blackwell weight layout takes its original early return.
            backend._low_ratio_write_group.reset_mock()
            layer.compressor.use_fused_compress = True
            backend._low_ratio_compress_decode(layer, x, Mock(), Mock())
            backend._low_ratio_compress_fused.assert_called_once()
            backend._low_ratio_write_group.assert_not_called()

    def test_cache_layout_and_tensor_contracts(self):
        def tensor(shape, dtype, **kwargs):
            return SimpleNamespace(
                shape=shape,
                ndim=len(shape),
                dtype=dtype,
                device="cuda:0",
                is_cuda=True,
                is_contiguous=lambda: True,
                **kwargs,
            )

        def inputs():
            return [
                tensor((6, 512), torch.float32),
                tensor((512,), torch.bfloat16),
                tensor((8192, 32), torch.complex64),
                tensor((6,), torch.int32),
                tensor((6,), torch.int64),
                tensor((4, 74880), torch.uint8),
                128,
            ]

        with patch.object(torch.version, "cuda", "12.8"):
            self.assertTrue(kernels.can_use_c2_epilogue(*inputs()))
            # Uniform FP8/BF16 storage, mismatched page size, strided projections,
            # wrong shapes/dtypes/devices and missing cache must all fall back.
            cases = [
                (5, "shape", (4, 65536)),
                (5, "dtype", torch.bfloat16),
                (0, "is_contiguous", lambda: False),
                (0, "is_cuda", False),
                (0, "dtype", torch.bfloat16),
                (1, "shape", (128,)),
                (2, "dtype", torch.complex128),
                (3, "shape", (5,)),
                (4, "dtype", torch.float32),
                (5, "device", "cuda:1"),
            ]
            for index, field, value in cases:
                args = inputs()
                setattr(args[index], field, value)
                with self.subTest(index=index, field=field):
                    self.assertFalse(kernels.can_use_c2_epilogue(*args))

            for page_size in (0, 64, 127):
                args = inputs()
                args[-1] = page_size
                self.assertFalse(kernels.can_use_c2_epilogue(*args))
            args = inputs()
            args[-2] = None
            self.assertFalse(kernels.can_use_c2_epilogue(*args))

    def test_verify_uses_pair_chain_only_for_a_safe_static_window(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(backend_module.DeepseekV4AttnBackend)
        backend.is_dspark_draft = False
        backend.speculative_num_draft_tokens = 6
        backend.token_to_kv_pool = Mock()
        backend.token_to_kv_pool.get_attention_compress_states.return_value = (
            SimpleNamespace(ring_size=8)
        )
        layer = SimpleNamespace(
            layer_id=2,
            compress_ratio=2,
            compressor=SimpleNamespace(use_fused_compress=False),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY, batch_size=2)
        with (
            backend_module.envs.SGLANG_OPT_DSV41_C2_EPILOGUE.override(True),
            patch.object(
                backend_module,
                "get_platform",
                return_value=SimpleNamespace(is_sm90=True),
            ),
            patch.object(backend_module, "read_ragged_verify_mode", return_value=None),
        ):
            self.assertTrue(
                backend._use_c2_verify_epilogue(layer, torch.empty(12, 1), batch)
            )
            self.assertFalse(
                backend._use_c2_verify_epilogue(layer, torch.empty(11, 1), batch)
            )
            backend.speculative_num_draft_tokens = 8
            self.assertFalse(
                backend._use_c2_verify_epilogue(layer, torch.empty(16, 1), batch)
            )
            backend.speculative_num_draft_tokens = 6
            with patch.object(
                backend_module,
                "read_ragged_verify_mode",
                return_value=backend_module.RaggedVerifyMode.COMPACT,
            ):
                self.assertFalse(
                    backend._use_c2_verify_epilogue(layer, torch.empty(12, 1), batch)
                )


if __name__ == "__main__":
    unittest.main()
