"""Candidate consumers follow source state across ratios and verify row chunks."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.v41_indexer.compact_decode import (
    CompactDecodeBlocks,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
MODULE = "sglang.srt.layers.attention.dsv4.v41_indexer.compact_decode"
DENSE = "sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks"


def fake_scores(
    q,
    weights,
    req,
    mapping,
    lens,
    table,
    page_size,
    lmax,
    ratio,
    *,
    candidate_blocks=None,
    candidate_block_size=8,
    return_block_scores=False,
):
    if candidate_blocks is None:
        logical = torch.arange(lmax).expand(q.shape[0], -1)
    else:
        logical = (
            candidate_blocks[:, :, None] * candidate_block_size
            + torch.arange(candidate_block_size)
        ).flatten(1)
    valid = torch.arange(logical.shape[1])[None, :] < lens[:, None]
    slots = mapping[req[:, None], logical.masked_fill(~valid, 0) * ratio] // ratio
    # Vary the ranking across rows/layers, without ties or shared request order.
    scores = ((slots * (17 + q[:, 0, 0, None].long()) + 13) % 101).float()
    scores = (scores + logical * 0.0001).masked_fill(~valid, -torch.inf)
    if not return_block_scores:
        return scores
    blocks = F.pad(
        scores, (0, -scores.shape[1] % candidate_block_size), value=-torch.inf
    )
    blocks = blocks.unflatten(-1, (-1, candidate_block_size)).amax(-1)
    blocks.masked_fill_(
        torch.arange(blocks.shape[1]) == (lens[:, None] - 1) // candidate_block_size,
        torch.inf,
    )
    return scores, blocks


def fake_topk(scores, lens, page_table, out, page_size, plan):
    scores = scores.masked_fill(
        torch.arange(scores.shape[1]) >= lens[:, None], -torch.inf
    )
    out.copy_(scores.topk(out.shape[1], sorted=False).indices)


class TestCompactDecode(CustomTestCase):
    def test_chunks_fit_aligned_score_and_source_block_allocations(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer.compact_decode import (
            _score_rows_per_chunk,
        )

        budget = 128 << 20
        reserve = 8 << 20
        for width in (259, 16384, 1048580):
            for block_size in (None, 1, 2, 8, 64):
                with self.subTest(width=width, block_size=block_size):
                    rows = _score_rows_per_chunk(width, block_size=block_size)
                    # Allocate actual padded row layouts, independently of the
                    # budget helper, so changing either alignment is caught.
                    logits = torch.empty(
                        rows, ((width + 31) // 32) * 32, dtype=torch.float32
                    )
                    nbytes = logits.numel() * logits.element_size()
                    if block_size is not None:
                        blocks = (width + block_size - 1) // block_size
                        maxima = torch.empty(
                            rows, ((blocks + 3) // 4) * 4, dtype=torch.float32
                        )
                        nbytes += maxima.numel() * maxima.element_size()
                    self.assertLessEqual(nbytes + reserve, budget)
        self.assertEqual(_score_rows_per_chunk(1048580, block_size=8), 26)

    def test_verify_rows_chunking_mixed_ratios_and_underfill(self):
        req = torch.tensor([2, 2, 0, 0, 0, 1])
        positions = torch.tensor([258, 259, -1, 0, 7, 16])
        mapping = torch.arange(3 * 1024).reshape(3, 1024).int()
        pool = Mock()
        pool.get_index_k_with_scale_buffer.return_value = torch.empty(1, 68)
        backend = DenseBlocksBackend(
            token_to_kv_pool=pool,
            req_to_token=mapping,
            candidate_topk_blocks=4,
            candidate_block_size=8,
            use_deep_gemm_prefill=False,
        )

        def inputs(ratio, capacity, seed):
            out = torch.full((6, 15), 999, dtype=torch.int32)
            indexer = SimpleNamespace(
                index_topk=9, queries=lambda q, f: q, head_weights=lambda x: x
            )
            return SimpleNamespace(
                req_rows=req,
                positions=positions,
                compress_ratio=ratio,
                layer_id=seed,
                indexer=indexer,
                q_lora=torch.full((6, 1, 1), seed),
                freqs_cis=torch.ones(300),
                x=torch.zeros(6, 1),
                out_page_indices=out,
                paged_metadata=SimpleNamespace(max_compressed_seq_len=capacity),
                reset_outputs=lambda: out.fill_(-1),
            )

        with (
            patch(f"{DENSE}.get_platform", return_value=SimpleNamespace(is_sm90=True)),
            patch(
                f"{MODULE}.fp4_index_logits_paged", side_effect=fake_scores
            ) as scorer,
            patch(
                f"{MODULE}.plan_topk_v2", side_effect=lambda lengths: lengths.clone()
            ),
            patch(f"{MODULE}.topk_transform_paged_v2", side_effect=fake_topk),
            patch(f"{MODULE}._SCORE_BUDGET_BYTES", 512 * 8 * 2),
        ):
            for source_ratio, consumer_ratio in ((1, 2), (2, 1)):
                source = inputs(source_ratio, 512, 3)
                published = backend.publish_decode(source)
                self.assertIsInstance(published, CompactDecodeBlocks)
                # For ratio 2 the consumer capacity is below its own publish gate.
                consumer = inputs(consumer_ratio, 512 // consumer_ratio, 7)
                backend.consume_decode(consumer, published)
                call = scorer.call_args
                self.assertIsNotNone(call.kwargs["candidate_blocks"])
                self.assertEqual(call.args[4].shape[0], 6)
                lengths = ((positions + 1) // consumer_ratio).clamp(min=0)
                logical = (published.blocks[:, :, None] * 8 + torch.arange(8)).flatten(
                    1
                )
                valid = logical < lengths[:, None]
                valid &= torch.arange(32)[None, :] < published.counts[:, None]
                safe = logical.masked_fill(~valid, 0)
                slots = mapping[req[:, None], safe * consumer_ratio] // consumer_ratio
                scores = ((slots * 24 + 13) % 101).float() + logical * 0.0001
                scores.masked_fill_(~valid, -torch.inf)
                chosen = scores.topk(9, sorted=False).indices
                chosen = logical.gather(1, chosen).masked_fill(
                    ~valid.gather(1, chosen), 1024
                )
                chosen = chosen.sort(-1).values
                expected = (
                    mapping[req[:, None], chosen.clamp_max(0x1FF) * consumer_ratio]
                    // consumer_ratio
                ).int()
                expected.masked_fill_(chosen == 1024, -1)
                torch.testing.assert_close(consumer.out_page_indices[:, :9], expected)
                self.assertTrue(torch.all(consumer.out_page_indices[:, 9:] == -1))
                self.assertTrue(torch.all(consumer.out_page_indices[2] == -1))

    def test_source_capacity_gate_keeps_dense_short_context_path(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer.compact_decode import (
            can_publish_compact,
        )

        inputs = SimpleNamespace(
            compress_ratio=1,
            indexer=SimpleNamespace(index_topk=2048),
            paged_metadata=SimpleNamespace(max_compressed_seq_len=262143),
        )
        mapping = torch.empty(1, 262144)
        self.assertFalse(
            can_publish_compact(inputs, mapping, topk_blocks=2048, block_size=8)
        )
        inputs.paged_metadata.max_compressed_seq_len = 262144
        self.assertTrue(
            can_publish_compact(inputs, mapping, topk_blocks=2048, block_size=8)
        )
        self.assertFalse(
            can_publish_compact(inputs, mapping, topk_blocks=2049, block_size=8)
        )


if __name__ == "__main__":
    unittest.main()
