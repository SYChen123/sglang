"""CP decode must slice every representation consumed by FP8 linears."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
CTX = "sglang.srt.layers.cp.cp_decode_attn_tp"


class TestCPFP8Views(CustomTestCase):
    def test_cached_bf16_and_grouped_scales_restore_after_repeated_decode(self):
        for cls, dim in ((ColumnParallelLinear, 0), (RowParallelLinear, 1)):
            full = torch.arange(64.0).reshape(8, 8)
            outputs = []
            for rank in range(2):
                layer = cls.__new__(cls)
                nn.Module.__init__(layer)
                layer.weight = nn.Parameter(full.clone(), requires_grad=False)
                layer._block_fp8_bf16_weight = full.clone()
                # Row scales can be [groups, output blocks, input blocks].
                scale = torch.arange(32.0).reshape(2, 4, 4) if dim else torch.ones(8, 2)
                layer.weight_scale_inv = nn.Parameter(scale, requires_grad=False)
                size_attr = (
                    "input_size_per_partition" if dim else "output_size_per_partition"
                )
                setattr(layer, size_attr, 8)
                layer.use_decode_attn_tp = False
                ctx = object.__new__(CpDecodeAttnTpContext)
                ctx.decode_tp_rank, ctx.decode_tp_size = rank, 2
                ctx._slice_cache = {}
                saved = [
                    layer.weight,
                    layer.weight_scale_inv,
                    layer._block_fp8_bf16_weight,
                ]
                pointers = [t.data_ptr() for t in saved]
                shard_pointer = None
                with (
                    patch(f"{CTX}.is_cp_active", return_value=False),
                    patch(f"{CTX}.dsa_use_prefill_cp", return_value=False),
                ):
                    for _ in range(2):
                        with ctx.maybe_use_decode_attn_tp(None, [layer]):
                            expected = full.chunk(2, dim=dim)[rank]
                            torch.testing.assert_close(layer.weight, expected)
                            torch.testing.assert_close(
                                layer._block_fp8_bf16_weight, expected
                            )
                            expected_scale = scale.chunk(2, dim=2 if dim else 0)[rank]
                            torch.testing.assert_close(
                                layer.weight_scale_inv, expected_scale
                            )
                            pointer = layer._block_fp8_bf16_weight.data_ptr()
                            if shard_pointer is not None:
                                self.assertEqual(pointer, shard_pointer)
                            shard_pointer = pointer
                            if dim:
                                outputs.append(
                                    torch.ones(1, 4) @ layer._block_fp8_bf16_weight.T
                                )
                            else:
                                outputs.append(
                                    torch.ones(1, 8) @ layer._block_fp8_bf16_weight.T
                                )
                        self.assertEqual(layer.weight.data_ptr(), pointers[0])
                        self.assertEqual(layer.weight_scale_inv.data_ptr(), pointers[1])
                        self.assertEqual(
                            layer._block_fp8_bf16_weight.data_ptr(), pointers[2]
                        )
                        self.assertEqual(getattr(layer, size_attr), 8)
                    with self.assertRaisesRegex(RuntimeError, "forward failed"):
                        with ctx.maybe_use_decode_attn_tp(None, [layer]):
                            raise RuntimeError("forward failed")
                    self.assertEqual(
                        layer._block_fp8_bf16_weight.data_ptr(), pointers[2]
                    )
            actual = sum(outputs[::2]) if dim else torch.cat(outputs[::2], dim=1)
            torch.testing.assert_close(actual, torch.ones(1, 8) @ full.T)

    def test_row_slice_keeps_scalar_and_per_output_channel_scales(self):
        layer = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(layer)
        layer.weight = nn.Parameter(torch.ones(8, 8))
        layer.weight_scale_inv = nn.Parameter(torch.ones(8))
        layer.weight_scale = nn.Parameter(torch.ones(()))
        ctx = object.__new__(CpDecodeAttnTpContext)
        attrs = ctx._get_linear_attrs(layer)
        self.assertEqual([(name, dim) for _, name, dim in attrs], [("weight", 1)])


if __name__ == "__main__":
    unittest.main()
