"""CP decode buckets must admit logical decode/verify/draft token counts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.padding import (
    get_cp_padding_align_size,
    should_unpad_cp_decode_attn_tp,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
PAD = "sglang.srt.layers.cp.padding"
UTIL = "sglang.srt.utils.common"
GRAPH = "sglang.srt.model_executor.runner.base_cuda_graph_runner"


class TestCPDecodeUnpadding(CustomTestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(
            enable_cp_decode_attn_tp=True,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=8,
            attn_tp_size=1,
        )
        self.runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(size=64),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
            ),
        )
        self.config = SimpleNamespace(
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(bs=[1, 2, 4, 8, 16, 64])
                ),
                torch_compile_max_bs=0,
            ),
        )

    def test_one_request_decode_verify_and_draft_buckets_survive(self):
        with (
            patch(f"{PAD}.get_parallel", return_value=self.parallel),
            patch(f"{UTIL}.get_parallel", return_value=self.parallel),
            patch(f"{UTIL}.get_exec", return_value=self.config),
            patch(f"{UTIL}.is_hip", return_value=False),
            patch(f"{UTIL}.require_gathered_buffer", return_value=False),
            patch(f"{GRAPH}.get_exec", return_value=self.config),
            patch(
                f"{GRAPH}.get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=False)
                ),
            ),
            patch("sglang.srt.layers.cp.base.is_zigzag", return_value=False),
            envs.SGLANG_CP_DISABLE_DECODE_UNPADDING.override(False),
        ):
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                self.assertEqual(get_cp_padding_align_size(mode, self.runner), 1)
            for width in (1, 5, 6):
                self.assertEqual(
                    get_batch_sizes_to_capture(self.runner, width)[0][0], 1
                )
            self.assertEqual(
                get_cp_padding_align_size(ForwardMode.EXTEND, self.runner), 8
            )
            self.assertEqual(get_cp_padding_align_size(), 8)
            with envs.SGLANG_CP_DISABLE_DECODE_UNPADDING.override(True):
                self.assertEqual(get_batch_sizes_to_capture(self.runner, 1)[0][0], 8)
                self.assertEqual(get_batch_sizes_to_capture(self.runner, 5)[0][0], 8)
                self.assertEqual(get_batch_sizes_to_capture(self.runner, 6)[0][0], 4)

    def test_topology_and_architecture_guards(self):
        with (
            patch(f"{PAD}.get_parallel", return_value=self.parallel),
            envs.SGLANG_CP_DISABLE_DECODE_UNPADDING.override(False),
        ):
            self.assertTrue(should_unpad_cp_decode_attn_tp(self.runner))
            for attr, value in (
                ("enable_cp_decode_attn_tp", False),
                ("enable_prefill_cp", False),
                ("cp_strategy", "zigzag"),
                ("dp_size", 2),
                ("attn_dp_size", 2),
                ("attn_cp_size", 1),
                ("attn_tp_size", 2),
            ):
                saved = getattr(self.parallel, attr)
                with self.subTest(attr=attr):
                    setattr(self.parallel, attr, value)
                    self.assertFalse(should_unpad_cp_decode_attn_tp(self.runner))
                setattr(self.parallel, attr, saved)
            self.runner.model_config.hf_config.architectures = ["LlamaForCausalLM"]
            self.assertFalse(should_unpad_cp_decode_attn_tp(self.runner))


if __name__ == "__main__":
    unittest.main()
