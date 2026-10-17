"""SM90 DG prefill: explicit FP8 arithmetic, causal cleaning and CP ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.sm90_deepgemm_prefill import (
    deepgemm_fp8_index_logits_prefill,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90 CUDA",
)
class TestSM90DeepGEMMPrefill(CustomTestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_fp64_reference_signed_weights_scales_odd_rows_and_causal_topk(self):
        for heads in (32, 64):
            for width in (3, 259):
                with self.subTest(heads=heads, width=width):
                    q = torch.randn(7, heads, 128, device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    keys = torch.randn(width, 128, device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    weights = torch.randn(7, heads, device="cuda").bfloat16()
                    scales = torch.rand(width, device="cuda") + 0.5
                    lens = torch.tensor(
                        [0, 1, 2, width - 1, width, 0, width],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    actual = deepgemm_fp8_index_logits_prefill(
                        q, weights, keys, lens, key_scales=scales
                    )
                    dot = torch.einsum("mhd,nd->mhn", q.double(), keys.double()).relu()
                    weighted = dot * weights.double()[:, :, None]
                    reference = weighted.sum(1) * scales.double()[None, :]
                    # Bound cancellation error by total absolute contribution,
                    # rather than a relative error that explodes near zero.
                    tolerance = (
                        weighted.abs().sum(1) * scales.double()[None, :] * 1e-3 + 1e-3
                    )
                    visible = (
                        torch.arange(width, device="cuda")[None, :] < lens[:, None]
                    )
                    self.assertTrue(
                        (
                            (actual.double() - reference).abs()[visible]
                            <= tolerance[visible]
                        ).all()
                    )
                    self.assertTrue(torch.isneginf(actual[~visible]).all())
                    self.assertEqual(actual.stride(0) % 4, 0)
                    k = min(5, width)
                    selected = torch.empty(7, k, device="cuda", dtype=torch.int32)
                    topk_transform_ragged_v2(
                        actual,
                        lens,
                        out_offsets=torch.zeros_like(lens),
                        out_indices=selected,
                    )
                    self.assertTrue(
                        torch.equal((selected >= 0).sum(1), lens.clamp(max=k).long())
                    )
                    self.assertTrue(((selected < lens[:, None]) | (selected < 0)).all())
                    for row in (1, 2, 3, 4, 6):
                        picked = selected[row][selected[row] >= 0].long()
                        expected = (
                            actual[row, : int(lens[row])]
                            .topk(min(k, int(lens[row])))
                            .values
                        )
                        torch.testing.assert_close(
                            actual[row, picked].sort().values, expected.sort().values
                        )

    def test_graph_replay_recleans_shrinking_live_lengths(self):
        q = torch.randn(5, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        keys = torch.randn(259, 128, device="cuda").to(torch.float8_e4m3fn)
        weights = torch.randn(5, 32, device="cuda").bfloat16()
        lens = torch.full((5,), 259, device="cuda", dtype=torch.int32)
        scales = torch.ones(259, device="cuda")
        for _ in range(2):
            deepgemm_fp8_index_logits_prefill(q, weights, keys, lens, key_scales=scales)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scores = deepgemm_fp8_index_logits_prefill(
                q, weights, keys, lens, key_scales=scales
            )
        for lengths in (
            [259, 259, 259, 259, 259],
            [0, 1, 3, 63, 0],
            [256, 0, 129, 1, 259],
        ):
            lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            visible = torch.arange(259, device="cuda")[None, :] < lens[:, None]
            self.assertTrue(torch.isneginf(scores[~visible]).all())
            expected = deepgemm_fp8_index_logits_prefill(
                q, weights, keys, lens, key_scales=scales
            )
            torch.testing.assert_close(
                scores[visible], expected[visible], rtol=0, atol=0
            )

    def test_runtime_cp_empty_shard_source_tail_consumer_and_padding(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import sm90_prefill

        # All FP4 codes are +/-1 with four different scales per row. This
        # exercises real gather/unpack without depending on a packer reference.
        page_size = 16
        table = torch.full(
            (256, page_size * 68), 0xAA, device="cuda", dtype=torch.uint8
        )
        table[:, page_size * 64 :] = torch.tensor(
            [125, 126, 127, 128], device="cuda", dtype=torch.uint8
        ).repeat(256, page_size)
        req_table = torch.arange(8 * 512, device="cuda").view(8, 512) * 2
        positions = torch.tensor([0, 4, 16, 0, 0], device="cuda")
        raw = torch.full((5, 16), 999, device="cuda", dtype=torch.int32)
        inputs = PrefillInputs(
            indexer=SimpleNamespace(
                index_topk=16,
                index_head_dim=128,
                n_local_heads=32,
                queries=lambda q, f: q,
                head_weights=lambda x: x,
            ),
            layer_id=0,
            compress_ratio=2,
            freqs_cis=torch.ones(512, 1, device="cuda"),
            x=torch.ones(5, 32, device="cuda", dtype=torch.bfloat16),
            q_lora=torch.randn(5, 32, 128, device="cuda").bfloat16(),
            positions=positions,
            req_rows=None,
            req_pool_indices=torch.tensor([1, 3, 5], device="cuda"),
            kv_page_table=torch.empty(0, device="cuda"),
            seq_lens_cpu=[33, 128, 17],
            rows_per_request=[2, 0, 1],
            rows_per_request_device=torch.tensor([2, 0, 1], device="cuda"),
            out_raw_indices=raw,
            out_page_indices=raw.clone(),
        )
        pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: table)
        backend = FullTopKIndexer(
            token_to_kv_pool=pool,
            req_to_token=req_table,
            use_deep_gemm_prefill=False,
            use_deep_gemm_decode=False,
        )
        for mode in ("deepgemm", "auto"):
            with (
                self.subTest(mode=mode),
                envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False),
                envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.override(mode),
                envs.SGLANG_DSV41_SM90_PREFILL_INDEXER_DEEPGEMM_MIN_ROWS.override(2),
                patch.object(sm90_prefill, "_SCORE_BUDGET_BYTES", 1024),
            ):
                backend.topk_prefill(inputs)
                for row, req in enumerate((1, 1, 5)):
                    visible = (int(positions[row]) + 1) // 2
                    self.assertEqual(set(raw[row].tolist()) - {-1}, set(range(visible)))
                    picked = raw[row][raw[row] >= 0].long()
                    self.assertTrue(
                        torch.equal(
                            inputs.out_page_indices[row, :visible].long(),
                            req_table[req, picked * 2] // 2,
                        )
                    )
                self.assertTrue((raw[3:] == -1).all())
                candidate = DenseBlocksBackend(
                    token_to_kv_pool=pool,
                    req_to_token=req_table,
                    candidate_topk_blocks=1,
                    candidate_block_size=8,
                    use_deep_gemm_prefill=False,
                )
                published = candidate.publish_prefill(inputs)
                self.assertEqual(published.rows_per_request, [2, 0, 1])
                self.assertEqual(published.tail([1, 0, 1]).blocks.shape[0], 2)
                published.blocks.fill_(-1)
                candidate.consume_prefill(inputs, published)
                self.assertTrue((raw == -1).all())
                self.assertTrue((inputs.out_page_indices == -1).all())
        with envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.override("invalid"):
            with self.assertRaisesRegex(ValueError, "triton.*deepgemm"):
                backend.topk_prefill(inputs)

    def test_auto_scores_actual_cp_tiles_and_preserves_forced_backends(self):
        """Budget-limited queries and the final tail must route independently."""
        from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
            fp8_index_logits_prefill,
            quantize_bf16_index_queries_fp8,
            unpack_fp4_index_keys_to_fp8,
        )
        from sglang.srt.layers.attention.dsv4.v41_indexer import sm90_prefill

        # Each entry gives the independently expected tile sizes and arithmetic
        # contract. In particular, a 256-row request with 67K K cannot fit one
        # 256-row DG tile under the production 64 MiB score budget.
        cases = [
            ("auto", 128, 127, 259, 64 << 20, [(127, "triton")]),
            ("auto", 128, 128, 259, 64 << 20, [(128, "deepgemm")]),
            ("auto", 128, 129, 259, 64 << 20, [(129, "deepgemm")]),
            ("auto", 256, 255, 259, 64 << 20, [(255, "triton")]),
            ("auto", 256, 256, 259, 64 << 20, [(256, "deepgemm")]),
            ("auto", 256, 257, 259, 64 << 20, [(257, "deepgemm")]),
            ("triton", 1, 256, 259, 64 << 20, [(256, "triton")]),
            ("deepgemm", 1024, 255, 259, 64 << 20, [(255, "deepgemm")]),
            (
                "auto",
                256,
                513,
                259,
                256 * 768 * 4,
                [(256, "deepgemm"), (256, "deepgemm"), (1, "triton")],
            ),
            ("auto", 256, 256, 67009, 64 << 20, [(248, "triton"), (8, "triton")]),
            ("auto", 128, 256, 67009, 64 << 20, [(248, "deepgemm"), (8, "triton")]),
        ]
        for mode, threshold, rows, width, budget, expected in cases:
            with self.subTest(mode=mode, threshold=threshold, rows=rows, width=width):
                page_size = 16
                pages = (width + page_size - 1) // page_size
                table = torch.randint(
                    0, 256, (pages, page_size * 68), device="cuda", dtype=torch.uint8
                )
                table[:, page_size * 64 :] = torch.tensor(
                    [125, 126, 127, 128], device="cuda", dtype=torch.uint8
                ).repeat(pages, page_size)
                raw = torch.full((rows + 2, 16), 999, device="cuda", dtype=torch.int32)
                q = torch.randn(rows + 2, 32, 128, device="cuda").bfloat16()
                weights = torch.randn(rows + 2, 32, device="cuda").bfloat16()
                positions = torch.full(
                    (rows + 2,), width - 1, device="cuda", dtype=torch.int64
                )
                positions[0] = 0
                inputs = PrefillInputs(
                    indexer=SimpleNamespace(
                        index_topk=16,
                        queries=lambda q, freqs: q,
                        head_weights=lambda x: x,
                    ),
                    layer_id=0,
                    compress_ratio=1,
                    freqs_cis=torch.ones(width, 1, device="cuda"),
                    x=weights,
                    q_lora=q,
                    positions=positions,
                    req_rows=None,
                    req_pool_indices=torch.tensor([1, 3], device="cuda"),
                    kv_page_table=torch.empty(0, device="cuda"),
                    seq_lens_cpu=[width, width],
                    rows_per_request=[0, rows],
                    rows_per_request_device=torch.tensor([0, rows], device="cuda"),
                    out_raw_indices=raw,
                    out_page_indices=raw.clone(),
                )
                req_table = torch.arange(width, device="cuda").repeat(4, 1)
                pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: table)
                keys = unpack_fp4_index_keys_to_fp8(
                    torch.arange(width, device="cuda"), table, page_size
                )
                scales = torch.ones(width, device="cuda", dtype=torch.float32)
                with (
                    envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.override(mode),
                    envs.SGLANG_DSV41_SM90_PREFILL_INDEXER_DEEPGEMM_MIN_ROWS.override(
                        threshold
                    ),
                    patch.object(sm90_prefill, "_SCORE_BUDGET_BYTES", budget),
                ):
                    chunks = []
                    for request, request_chunks in sm90_prefill.prefill_requests(
                        inputs, pool, req_table
                    ):
                        chunks.extend(request_chunks)
                    self.assertEqual(
                        [len(c.tok) for c in chunks], [n for n, _ in expected]
                    )
                    for chunk, (_, selected) in zip(chunks, expected):
                        q8 = quantize_bf16_index_queries_fp8(q[chunk.tok])
                        if selected == "deepgemm":
                            reference = deepgemm_fp8_index_logits_prefill(
                                q8,
                                weights[chunk.tok],
                                keys,
                                chunk.lens,
                                key_scales=scales,
                            )
                        else:
                            reference = fp8_index_logits_prefill(
                                q8, weights[chunk.tok], keys, chunk.lens
                            )
                        torch.testing.assert_close(
                            chunk.scores, reference, rtol=0, atol=0
                        )
                        self.assertEqual(chunk.scores.stride(0) % 4, 0)
                    self.assertTrue((raw == -1).all())
                for invalid_threshold in (0, -1):
                    with (
                        envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.override("auto"),
                        envs.SGLANG_DSV41_SM90_PREFILL_INDEXER_DEEPGEMM_MIN_ROWS.override(
                            invalid_threshold
                        ),
                        self.assertRaisesRegex(ValueError, "MIN_ROWS must be positive"),
                    ):
                        list(sm90_prefill.prefill_requests(inputs, pool, req_table))


if __name__ == "__main__":
    unittest.main()
