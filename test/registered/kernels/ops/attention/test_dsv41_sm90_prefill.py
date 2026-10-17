"""Hopper prefill preserves FP4 layout, BF16 rounding and CP row ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
    fp8_index_logits_prefill,
    quantize_bf16_index_queries_fp8,
    unpack_fp4_index_keys_to_fp8,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    BlockIds,
    DenseBlocksBackend,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def packed_keys(num_slots, page_size=16):
    """Independent E2M1 lookup reference, with a padded physical page stride."""
    pages = (num_slots + page_size - 1) // page_size
    backing = torch.zeros(pages, page_size * 68 + 16, device="cuda", dtype=torch.uint8)
    table = backing[:, : page_size * 68]
    payload = torch.randint(
        0, 256, (pages, page_size, 64), device="cuda", dtype=torch.uint8
    )
    exps = torch.randint(
        120, 131, (pages, page_size, 4), device="cuda", dtype=torch.uint8
    )
    table[:, : page_size * 64] = payload.flatten(1)
    table[:, page_size * 64 :] = exps.flatten(1)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack((payload & 15, payload >> 4), -1).flatten(-2).long()
    values = levels[codes] * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)
    return table, values.flatten(0, 1).to(torch.bfloat16)


def rounded_scores(q, keys, weights):
    scores = torch.einsum("mhd,nd->mhn", q, keys)
    return (scores.relu() * weights[:, :, None]).sum(1).float()


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90 CUDA",
)
class TestSM90Prefill(CustomTestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_gather_padded_pages_repeated_slots_and_cast_limits(self):
        table, values = packed_keys(65)
        slots = torch.tensor([64, 0, 17, 17, 31], device="cuda")
        actual = unpack_fp4_index_keys_to_fp8(slots, table, 16)
        self.assertTrue(torch.equal(actual.float(), values[slots].float()))
        q = torch.tensor([-1000, -448, -0.001, 0, 448, 1000], device="cuda")
        q = q.repeat(128)[:128].reshape(1, 1, 128).repeat(2, 32, 1).bfloat16()
        actual = quantize_bf16_index_queries_fp8(q)
        expected = q.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(actual.float(), expected.float()))

    def test_scores_rounding_signed_weights_odd_width_and_empty_rows(self):
        for heads in (32, 64):
            with self.subTest(heads=heads):
                table, values = packed_keys(259)
                slots = torch.arange(259, device="cuda").flip(0)
                keys = unpack_fp4_index_keys_to_fp8(slots, table, 16)
                q = values[torch.randint(0, 259, (6, heads), device="cuda")]
                q8 = quantize_bf16_index_queries_fp8(q)
                weights = torch.randn(6, heads, device="cuda").bfloat16()
                lens = torch.tensor(
                    [0, 1, 3, 63, 258, 259], device="cuda", dtype=torch.int32
                )
                actual = fp8_index_logits_prefill(q8, weights, keys, lens)
                ref = rounded_scores(q8.bfloat16(), keys.bfloat16(), weights)
                visible = torch.arange(259, device="cuda")[None, :] < lens[:, None]
                torch.testing.assert_close(
                    actual[visible], ref[visible], rtol=0, atol=0
                )
                self.assertTrue(torch.isneginf(actual[~visible]).all())
                self.assertEqual(actual.stride(0) % 4, 0)
                selected = torch.empty(6, 7, device="cuda", dtype=torch.int32)
                topk_transform_ragged_v2(
                    actual,
                    lens,
                    out_offsets=torch.zeros_like(lens),
                    out_indices=selected,
                )
                self.assertTrue(
                    torch.equal((selected >= 0).sum(1), lens.clamp(max=7).long())
                )
                self.assertTrue(((selected < lens[:, None]) | (selected < 0)).all())

    def test_runtime_cp_empty_shard_padding_multi_chunk_and_candidate_underfill(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import sm90_prefill

        table, values = packed_keys(10 * 512)
        pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: table)
        for ratio in (1, 2):
            with (
                self.subTest(ratio=ratio),
                envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False),
            ):
                req_table = torch.arange(10 * 512, device="cuda").view(10, 512) * ratio
                positions = torch.tensor([0, 4, 256, 2, 32, 0, 0], device="cuda")
                q = values[torch.randint(0, 100, (7, 32), device="cuda")]
                weights = torch.randn(7, 32, device="cuda").bfloat16()
                raw = torch.full((7, 16), 999, dtype=torch.int32, device="cuda")
                inputs = PrefillInputs(
                    indexer=SimpleNamespace(
                        index_topk=16,
                        index_head_dim=128,
                        n_local_heads=32,
                        queries=lambda q, freqs: q,
                        head_weights=lambda x: x,
                    ),
                    layer_id=0,
                    compress_ratio=ratio,
                    freqs_cis=torch.ones(512, 1, device="cuda"),
                    x=weights,
                    q_lora=q,
                    positions=positions,
                    req_rows=None,
                    req_pool_indices=torch.tensor([2, 6, 8], device="cuda"),
                    kv_page_table=torch.empty(0, device="cuda"),
                    seq_lens_cpu=[257, 128, 33],
                    rows_per_request=[3, 0, 2],
                    rows_per_request_device=torch.tensor([3, 0, 2], device="cuda"),
                    out_raw_indices=raw,
                    out_page_indices=raw.clone(),
                )
                backend = FullTopKIndexer(
                    token_to_kv_pool=pool,
                    req_to_token=req_table,
                    use_deep_gemm_prefill=False,
                    use_deep_gemm_decode=False,
                )
                # Force multiple logits chunks without reducing model chunk size.
                with patch.object(sm90_prefill, "_SCORE_BUDGET_BYTES", 1024):
                    backend.topk_prefill(inputs)
                for row, request_id in enumerate((2, 2, 2, 8, 8)):
                    visible = (int(positions[row]) + 1) // ratio
                    picked = raw[row][raw[row] >= 0].long()
                    self.assertEqual(picked.numel(), min(16, visible))
                    self.assertTrue((picked < visible).all())
                    slots = req_table[request_id, picked * ratio] // ratio
                    self.assertTrue(
                        torch.equal(
                            inputs.out_page_indices[row, : picked.numel()].long(), slots
                        )
                    )
                    if visible > 16:
                        all_slots = (
                            req_table[
                                request_id, torch.arange(visible, device="cuda") * ratio
                            ]
                            // ratio
                        )
                        ref = rounded_scores(
                            q[row : row + 1], values[all_slots], weights[row : row + 1]
                        )[0]
                        threshold = ref.topk(16).values[-1]
                        self.assertTrue((ref[picked] >= threshold).all())
                self.assertTrue((raw[5:] == -1).all())
                self.assertTrue((inputs.out_page_indices[5:] == -1).all())

                candidate = DenseBlocksBackend(
                    token_to_kv_pool=pool,
                    req_to_token=req_table,
                    candidate_topk_blocks=1,
                    candidate_block_size=8,
                    use_deep_gemm_prefill=False,
                )
                blocks = torch.zeros(7, 1, device="cuda", dtype=torch.int32)
                blocks[1] = -1  # all candidates masked despite positive causal length
                candidate.consume_prefill(inputs, BlockIds(blocks=blocks))
                self.assertTrue((raw[1] == -1).all())
                self.assertTrue((inputs.out_page_indices[1] == -1).all())
                for row in (0, 2, 3, 4):
                    visible = min(8, (int(positions[row]) + 1) // ratio)
                    self.assertEqual(set(raw[row].tolist()) - {-1}, set(range(visible)))


if __name__ == "__main__":
    unittest.main()
