"""Hopper prefill preserves FP4 layout, BF16 rounding and CP row ownership."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
    quantize_bf16_index_queries_fp8,
    unpack_fp4_index_keys_to_fp8,
)
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

    def test_runtime_cp_empty_shard_padding_multi_chunk_and_candidate_underfill(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        table, values = packed_keys(10 * 512)
        pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: table)
        for ratio in (1, 2):
            with (
                self.subTest(ratio=ratio),
                envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(True),
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
                        scores=rounded_scores,
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
                with (
                    envs.SGLANG_OPT_DSV41_LARGE_INDEXER_SCORE_BUDGET.override(False),
                    envs.SGLANG_OPT_DSV41_LARGE_PREFILL_SCORE_BUDGET.override(False),
                    patch.object(scoring, "_TORCH_SCORE_BUDGET_BYTES", 1024),
                ):
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

    def test_bf16_compact_scores_per_query_masks_topk_and_graph_replay(self):
        """Torch fallback scores only each row's candidates, including TP4 heads."""
        from sglang.kernels.ops.attention.dsv4.candidate_scores import (
            candidate_scores,
            candidate_topk,
        )

        for heads in (8, 32, 64):
            with self.subTest(heads=heads):
                q = torch.randn(5, heads, 128, device="cuda").bfloat16()
                keys = torch.randn(259, 128, device="cuda").bfloat16()
                weights = torch.randn(5, heads, device="cuda").bfloat16()
                lens = torch.tensor(
                    [0, 1, 63, 258, 259], device="cuda", dtype=torch.int32
                )
                blocks = torch.tensor(
                    [[0, -1, 1], [0, 2, -1], [7, -1, 0], [32, 1, 9], [3, 32, 100]],
                    device="cuda",
                    dtype=torch.int32,
                )
                for _ in range(2):
                    candidate_scores(q, keys, weights, blocks, lens)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = candidate_scores(q, keys, weights, blocks, lens)
                    picked = candidate_topk(actual, blocks, 32)
                for replay in range(2):
                    if replay:
                        blocks.copy_(blocks.flip(0))
                        lens.copy_(torch.tensor([259, 63, 0, 1, 258], device="cuda"))
                    graph.replay()
                    logical = (
                        blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
                    ).flatten(1)
                    valid = (blocks >= 0).repeat_interleave(8, 1)
                    valid &= (logical < lens[:, None]) & (logical < keys.shape[0])
                    # The Triton dot accumulates in FP32. Small cuBLAS BF16
                    # GEMMs may reduce in BF16; use an independent FP32 dot
                    # and retain both BF16 rounding boundaries of the epilogue.
                    dots = torch.einsum(
                        "bhd,nd->bhn", q.float(), keys.float()
                    ).bfloat16()
                    dense = (dots.relu() * weights[:, :, None]).sum(1).float()
                    expected = dense.gather(1, logical.clamp(0, 258)).masked_fill(
                        ~valid, -torch.inf
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for row in range(5):
                        self.assertEqual(
                            set(picked[row].tolist()) - {-1},
                            set(logical[row][valid[row]].tolist()),
                        )

    def test_bf16_head_epilogue_matches_torch_rounding_and_strides(self):
        """Fusion retains BF16 product rounding and the 32-head reduction order."""
        from sglang.kernels.ops.attention.dsv4.index_score_reduce import (
            reduce_head_scores,
        )

        for heads, width in ((32, 259), (32, 1088), (32, 1), (8, 129)):
            with self.subTest(heads=heads, width=width):
                backing = torch.randn(3, heads, width + 7, device="cuda").bfloat16()
                scores = backing[:, :, :width]
                weights = torch.randn(3, heads * 2, device="cuda").bfloat16()[:, ::2]
                expected = (scores.relu() * weights.unsqueeze(-1)).sum(1).float()
                actual = reduce_head_scores(scores, weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_torch_aligned_width_keeps_cp_causality_and_pool_mapping(self):
        """Padding a 1025-column GEMM must not publish its extra K columns."""
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        for heads in (8, 32):
            with self.subTest(heads=heads):
                _, values = packed_keys(4096)
                # Give each request 2048 logical slots, with nonidentity pool mapping.
                req_table = torch.randperm(4096, device="cuda").view(2, 2048)
                pool = SimpleNamespace(
                    get_low_ratio_index_k_dequant=lambda _, slots: values[slots]
                )
                positions = torch.tensor([16, 1024, 0], device="cuda")
                q = values[torch.randint(0, 100, (3, heads), device="cuda")]
                weights = torch.randn(3, heads, device="cuda").bfloat16()
                raw = torch.full((3, 16), 999, dtype=torch.int32, device="cuda")
                inputs = PrefillInputs(
                    indexer=SimpleNamespace(
                        index_topk=16,
                        index_head_dim=128,
                        n_local_heads=heads,
                        queries=lambda q, freqs: q,
                        head_weights=lambda x: x,
                        scores=rounded_scores,
                    ),
                    layer_id=0,
                    compress_ratio=1,
                    freqs_cis=torch.ones(2048, 1, device="cuda"),
                    x=weights,
                    q_lora=q,
                    positions=positions,
                    req_rows=None,
                    req_pool_indices=torch.tensor([1, 0], device="cuda"),
                    kv_page_table=torch.empty(0, device="cuda"),
                    seq_lens_cpu=[1025, 1],
                    rows_per_request=[2, 1],
                    rows_per_request_device=torch.tensor([2, 1], device="cuda"),
                    out_raw_indices=raw,
                    out_page_indices=raw.clone(),
                )
                backend = FullTopKIndexer(
                    token_to_kv_pool=pool,
                    req_to_token=req_table,
                    use_deep_gemm_prefill=False,
                    use_deep_gemm_decode=False,
                )
                with (
                    envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(True),
                    patch.object(
                        scoring, "_TORCH_SCORE_BUDGET_BYTES", heads * 1088 * 2
                    ),
                ):
                    backend.topk_prefill(inputs)
                for row, req in enumerate((1, 1, 0)):
                    visible = int(positions[row]) + 1
                    picked = raw[row][raw[row] >= 0].long()
                    self.assertEqual(picked.numel(), min(16, visible))
                    self.assertTrue((picked < visible).all())
                    keys = values[req_table[req, :visible]]
                    reference = rounded_scores(
                        q[row : row + 1], keys, weights[row : row + 1]
                    )[0]
                    threshold = reference.topk(min(16, visible)).values[-1]
                    self.assertTrue((reference[picked] >= threshold).all())
                    self.assertTrue(
                        torch.equal(
                            inputs.out_page_indices[row, : picked.numel()].long(),
                            req_table[req, picked],
                        )
                    )

    def test_forced_torch_scores_skip_fused_head_epilogue(self):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer

        q = torch.randn(3, 32, 128, device="cuda").bfloat16()
        keys = torch.randn(259, 128, device="cuda").bfloat16()
        weights = torch.randn(3, 32, device="cuda").bfloat16()
        with (
            envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(True),
            patch(
                "sglang.kernels.ops.attention.dsv4.index_score_reduce.reduce_head_scores",
                side_effect=AssertionError(
                    "Forced Torch must use the reference epilogue"
                ),
            ),
        ):
            actual = DeepseekV41Indexer.scores(None, q, keys, weights)
        torch.testing.assert_close(
            actual, rounded_scores(q, keys, weights), rtol=0, atol=0
        )

    def test_compact_cp_tail_maps_distinct_candidates_to_pool_slots(self):
        """A bounded CP tail uses its published rows, never padded encoder rows."""
        table, values = packed_keys(4 * 512)
        req_table = torch.randperm(4 * 512, device="cuda").view(4, 512)
        pool = SimpleNamespace(
            get_index_k_with_scale_buffer=lambda _: table,
            get_low_ratio_index_k_dequant=lambda _, slots: values[slots],
        )
        raw = torch.full((5, 16), 999, dtype=torch.int32, device="cuda")
        positions = torch.tensor([80, 90, 258, 0, 0], device="cuda")
        inputs = PrefillInputs(
            indexer=SimpleNamespace(
                index_topk=16,
                index_head_dim=128,
                n_local_heads=32,
                queries=lambda q, freqs: q,
                head_weights=lambda x: x,
                scores=rounded_scores,
            ),
            layer_id=0,
            compress_ratio=1,
            freqs_cis=torch.ones(512, 1, device="cuda"),
            x=torch.randn(5, 32, device="cuda").bfloat16(),
            q_lora=values[torch.randint(0, 100, (5, 32), device="cuda")],
            positions=positions,
            req_rows=None,
            req_pool_indices=torch.tensor([3, 1, 2], device="cuda"),
            kv_page_table=torch.empty(0, device="cuda"),
            seq_lens_cpu=[91, 128, 259],
            rows_per_request=[2, 0, 1],
            rows_per_request_device=torch.tensor([2, 0, 1], device="cuda"),
            out_raw_indices=raw,
            out_page_indices=raw.clone(),
        )
        # Encoder rows per request were [3,0,2]; only their tails reach decoder.
        published = BlockIds(
            blocks=torch.tensor(
                [[9, 10], [5, -1], [2, 7], [1, -1], [32, 3]],
                device="cuda",
                dtype=torch.int32,
            ),
            rows_per_request=[3, 0, 2],
        ).tail([2, 0, 1])
        backend = DenseBlocksBackend(
            token_to_kv_pool=pool,
            req_to_token=req_table,
            candidate_topk_blocks=2,
            candidate_block_size=8,
            use_deep_gemm_prefill=False,
        )
        for mode in ("torch", "deepgemm"):
            with (
                envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(mode == "torch"),
                envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.override("deepgemm"),
                (
                    patch(
                        "sglang.kernels.ops.attention.dsv4.candidate_scores.candidate_scores",
                        side_effect=AssertionError(
                            "Forced Torch must score the full request"
                        ),
                    )
                    if mode == "torch"
                    else nullcontext()
                ),
            ):
                backend.consume_prefill(inputs, published)
                for row, req in enumerate((3, 3, 2)):
                    candidates = {
                        block * 8 + offset
                        for block in published.blocks[row].tolist()
                        for offset in range(8)
                        if block >= 0 and block * 8 + offset <= int(positions[row])
                    }
                    picked = raw[row][raw[row] >= 0].long()
                    self.assertEqual(set(picked.tolist()), candidates)
                    self.assertTrue(
                        torch.equal(
                            inputs.out_page_indices[row, : picked.numel()].long(),
                            req_table[req, picked],
                        )
                    )
                self.assertTrue((raw[3:] == -1).all())
                self.assertTrue((inputs.out_page_indices[3:] == -1).all())


if __name__ == "__main__":
    unittest.main()
