"""Numerics and live-length CUDA graph replay for Hopper candidate indexing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_index_logits_decode as dense_logits,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_index_logits_paged,
)
from sglang.kernels.ops.attention.dsv4.sm90_candidate import (
    candidate_block_state,
    candidate_position_counts,
    finalize_candidate_topk,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def fp4_index_logits_decode(
    q,
    weights,
    mapping,
    lens,
    table,
    page_size,
    *,
    req_indices=None,
    compress_ratio=1,
    max_seq_len=None,
    bounded_grid=False,
    prefix_only=True,
    candidate_blocks=None,
    candidate_block_size=8,
    return_block_scores=False,
    skip_invisible=False,
):
    # Adapt the theta test fixture to this branch's separate dense/paged APIs.
    if req_indices is None:
        return dense_logits(q, weights, mapping, lens, table, page_size)
    return fp4_index_logits_paged(
        q,
        weights,
        req_indices,
        mapping,
        lens,
        table,
        page_size,
        max_seq_len,
        compress_ratio,
        candidate_blocks=candidate_blocks,
        candidate_block_size=candidate_block_size,
        return_block_scores=return_block_scores,
    )


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.cuda.get_device_capability()[0] >= 9,
    "requires SM90+ CUDA",
)
class TestSM90CandidateKernels(CustomTestCase):
    def test_runtime_backend_graph_replays_mixed_ratio_and_row_chunks(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
            DenseBlocksBackend,
        )

        rows, capacity = 6, 2048
        req = torch.tensor([2, 2, 0, 0, 0, 1], device="cuda")
        pos = torch.tensor([2047, 1023, 0, 1, 7, 16], device="cuda")
        mapping = torch.arange(3 * capacity, device="cuda", dtype=torch.int32).reshape(
            3, capacity
        )
        table = torch.randint(0, 256, (96, 64 * 68), device="cuda", dtype=torch.uint8)
        table[:, 64 * 64 :] = 127
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
        pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda layer: table)
        backend = DenseBlocksBackend(
            token_to_kv_pool=pool,
            req_to_token=mapping,
            candidate_topk_blocks=4,
            candidate_block_size=8,
            use_deep_gemm_prefill=False,
        )

        def make_inputs(ratio):
            out = torch.empty(rows, 16, device="cuda", dtype=torch.int32)
            return SimpleNamespace(
                req_rows=req,
                positions=pos,
                compress_ratio=ratio,
                layer_id=ratio,
                indexer=SimpleNamespace(
                    index_topk=9, queries=lambda q, f: q, head_weights=lambda x: x
                ),
                x=weights,
                q_lora=q,
                freqs_cis=torch.ones(capacity, device="cuda"),
                paged_metadata=SimpleNamespace(
                    max_compressed_seq_len=capacity // ratio
                ),
                out_page_indices=out,
                reset_outputs=lambda: out.fill_(-1),
            )

        source, consumer = make_inputs(1), make_inputs(2)
        module = "sglang.srt.layers.attention.dsv4.v41_indexer"
        with (
            patch(
                module + ".dense_blocks.get_platform",
                return_value=SimpleNamespace(is_sm90=True),
            ),
            patch(module + ".compact_decode._SCORE_BUDGET_BYTES", capacity * 8 * 2),
        ):

            def run():
                published = backend.publish_decode(source)
                backend.consume_decode(consumer, published)

            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for live in ([2048, 1024, 0, 1, 8, 17], [0] * rows, [2, 3, 4, 5, 6, 7]):
                pos.copy_(torch.tensor(live, device="cuda") - 1)
                req.copy_(req.roll(1))
                q.normal_()
                graph.replay()
                captured = consumer.out_page_indices.clone()
                run()
                torch.testing.assert_close(
                    captured, consumer.out_page_indices, rtol=0, atol=0
                )

    def test_full_and_compact_logits_graph_replay(self):
        for width in (259, 32771):
            with self.subTest(width=width, bounded_grid=True):
                self._check_graph_replay(width=width, bounded_grid=True)

    def _check_graph_replay(
        self, *, width, bounded_grid, prefix_only=True, topk=9, topk_blocks=7
    ):
        torch.manual_seed(17)
        rows, page_size = 6, 64
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
        req = torch.tensor([2, 2, 0, 0, 0, 1], device="cuda", dtype=torch.int64)
        lens = torch.tensor([259, 130, 0, 1, 8, 17], device="cuda", dtype=torch.int32)
        table = torch.randint(
            0, 256, (32, page_size * 68), device="cuda", dtype=torch.uint8
        )
        table[:, page_size * 64 :] = 127
        for ratio in (1, 2):
            mapping = torch.stack(
                [torch.randint(0, 2048, (width,), device="cuda") for _ in range(3)]
            ).int()
            mapping = mapping.repeat_interleave(ratio, 1) * ratio
            logical = torch.arange(width, device="cuda")
            explicit = mapping[req[:, None], logical * ratio] // ratio

            def run(mapping=mapping, ratio=ratio):
                scores, block_scores = fp4_index_logits_decode(
                    q,
                    weights,
                    mapping,
                    lens,
                    table,
                    page_size,
                    req_indices=req,
                    compress_ratio=ratio,
                    max_seq_len=width,
                    bounded_grid=bounded_grid,
                    prefix_only=prefix_only,
                    candidate_block_size=8,
                    return_block_scores=True,
                )
                source_selected = torch.empty(
                    (rows, topk), device="cuda", dtype=torch.int32
                )
                topk_transform_paged_v2(
                    scores, lens, None, source_selected, 1, plan_topk_v2(lens)
                )
                source_pages = torch.empty_like(source_selected)
                source_raw = torch.empty_like(source_selected)
                finalize_candidate_topk(
                    source_selected,
                    scores,
                    lens,
                    mapping,
                    req,
                    source_pages,
                    source_raw,
                    ratio=ratio,
                )
                blocks, counts = candidate_block_state(
                    block_scores, lens, topk_blocks=topk_blocks, block_size=8
                )
                compact = fp4_index_logits_decode(
                    q,
                    weights,
                    mapping,
                    counts,
                    table,
                    page_size,
                    req_indices=req,
                    compress_ratio=ratio,
                    max_seq_len=width,
                    bounded_grid=bounded_grid and not prefix_only,
                    prefix_only=prefix_only,
                    candidate_blocks=blocks,
                    candidate_block_size=8,
                )
                selected = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
                topk_transform_paged_v2(
                    compact, counts, None, selected, 1, plan_topk_v2(counts)
                )
                pages = torch.empty((rows, 2 * topk), device="cuda", dtype=torch.int32)[
                    :, :topk
                ]
                raw = torch.empty(
                    (rows, 2 * topk + 2), device="cuda", dtype=torch.int32
                )[:, :topk]
                finalize_candidate_topk(
                    selected,
                    compact,
                    counts,
                    mapping,
                    req,
                    pages,
                    raw,
                    ratio=ratio,
                    candidate_blocks=blocks,
                    candidate_block_size=8,
                )
                # Exercise mixed-ratio counts with a consumer seeing a longer context.
                expanded = candidate_position_counts(
                    blocks, counts, lens * 2, block_size=8
                )
                return (
                    scores,
                    block_scores,
                    blocks,
                    counts,
                    compact,
                    pages,
                    raw,
                    expanded,
                    source_pages,
                    source_raw,
                )

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            for live in (
                [width, width - 1, 0, 1, 8, 17],
                [0, 1, 2, 7, 8, 9],
                [width] * rows,
                [0] * rows,
            ):
                lens.copy_(torch.tensor(live, device="cuda"))
                q.normal_()
                weights.normal_()
                req.copy_(req.roll(1))
                explicit = mapping[req[:, None], logical * ratio] // ratio
                # Poison allocations left over from capture/previous replay. Length-
                # aware consumers must not select these NaNs or stale long tails.
                if prefix_only:
                    for scratch in (out[0], out[1], out[4]):
                        scratch.fill_(torch.nan)
                graph.replay()
                (
                    scores,
                    block_scores,
                    blocks,
                    counts,
                    compact,
                    pages,
                    raw,
                    expanded,
                    source_pages,
                    source_raw,
                ) = out
                expected = fp4_index_logits_decode(
                    q, weights, explicit, lens, table, page_size, skip_invisible=False
                )
                visible = logical[None, :] < lens[:, None]
                torch.testing.assert_close(
                    scores.masked_fill(~visible, -torch.inf) if prefix_only else scores,
                    expected,
                    rtol=0,
                    atol=0,
                )
                source_valid = source_raw >= 0
                torch.testing.assert_close(
                    source_valid.sum(-1), lens.clamp(max=topk).long()
                )
                source_values = scores.gather(
                    1, source_raw.long().clamp_min(0)
                ).masked_fill(~source_valid, -torch.inf)
                torch.testing.assert_close(
                    source_values.sort(-1).values,
                    expected.topk(topk).values.sort(-1).values,
                    rtol=0,
                    atol=0,
                )
                source_expected_pages = (
                    mapping[req[:, None], source_raw.long().clamp_min(0) * ratio]
                    // ratio
                ).masked_fill(~source_valid, -1)
                torch.testing.assert_close(source_pages, source_expected_pages.int())
                expected_blocks = (
                    F.pad(expected, (0, -width % 8), value=-torch.inf)
                    .unflatten(-1, (-1, 8))
                    .amax(-1)
                )
                expected_blocks.masked_fill_(
                    torch.arange(expected_blocks.shape[1], device="cuda")
                    == (lens[:, None] - 1) // 8,
                    torch.inf,
                )
                torch.testing.assert_close(
                    block_scores.masked_fill(
                        torch.arange(block_scores.shape[1], device="cuda")
                        >= ((lens[:, None] + 7) // 8),
                        -torch.inf,
                    )
                    if prefix_only
                    else block_scores,
                    expected_blocks,
                    rtol=0,
                    atol=0,
                )
                # Top-K ties may choose different blocks; compare their selected scores.
                valid_blocks = blocks < block_scores.shape[1]
                selected_scores = block_scores.gather(
                    1, blocks.long().clamp_max(block_scores.shape[1] - 1)
                ).masked_fill(~valid_blocks, -torch.inf)
                torch.testing.assert_close(
                    selected_scores.sort(-1).values,
                    expected_blocks.topk(topk_blocks).values.sort(-1).values,
                )
                positions = (
                    blocks[:, :, None] * 8 + torch.arange(8, device="cuda")
                ).flatten(1)
                valid = (
                    torch.arange(positions.shape[1], device="cuda") < counts[:, None]
                )
                expected_compact = expected.gather(
                    1, positions.long().clamp_max(width - 1)
                ).masked_fill(~valid, -torch.inf)
                torch.testing.assert_close(
                    compact.masked_fill(~valid, -torch.inf) if prefix_only else compact,
                    expected_compact,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(expanded, counts)
                visible = raw >= 0
                self.assertTrue(
                    torch.all(
                        raw.masked_fill(~visible, width).sort(-1).values
                        == raw.masked_fill(~visible, width)
                    )
                )
                selected_scores = scores.gather(1, raw.long().clamp_min(0)).masked_fill(
                    ~visible, -torch.inf
                )
                torch.testing.assert_close(
                    selected_scores.sort(-1).values,
                    expected_compact.topk(topk).values.sort(-1).values,
                )
                expected_pages = (
                    mapping[req[:, None], raw.long().clamp_min(0) * ratio] // ratio
                ).masked_fill(~visible, -1)
                torch.testing.assert_close(pages, expected_pages.int())
                self.assertEqual(scores.stride(0) % 4, 0)
                self.assertEqual(block_scores.stride(0) % 4, 0)

    def test_finalize_short_rows_nan_and_padding(self):
        scores = torch.tensor(
            [[3.0, torch.nan, -torch.inf, torch.inf, 1.0], [0.0] * 5], device="cuda"
        )
        selected = torch.tensor(
            [[1, 4, 3, 0, 2], [-1] * 5], device="cuda", dtype=torch.int32
        )
        lens = torch.tensor([5, 0], device="cuda", dtype=torch.int32)
        mapping = torch.arange(32, device="cuda").reshape(2, 16)
        pages = torch.full((2, 18), 77, device="cuda", dtype=torch.int32)
        raw = torch.full((2, 20), 88, device="cuda", dtype=torch.int32)
        finalize_candidate_topk(
            selected,
            scores,
            lens,
            mapping,
            torch.arange(2, device="cuda"),
            pages[:, :9],
            raw[:, :9],
            ratio=2,
        )
        expected = torch.tensor(
            [[0, 3, 4] + [-1] * 6, [-1] * 9], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(pages[:, :9], expected)
        torch.testing.assert_close(raw[:, :9], expected)
        self.assertTrue(torch.all(pages[:, 9:] == 77))
        self.assertTrue(torch.all(raw[:, 9:] == 88))


if __name__ == "__main__":
    unittest.main()
