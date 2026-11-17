"""Query DP preserves global ordering, ghost ranks and page-table row ownership."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.layers.cp.cp_decode_attn_dp import (
    CPDecodeQueryLayout,
    cp_decode_query_layout,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCPDecodeDP(CustomTestCase):
    def test_uneven_verify_and_ghost_shards_round_trip_without_reordering(self):
        for size in (2, 4, 8):
            for rows in (0, 1, 5, 6, 17, 144, 384):
                source = torch.arange(rows * 3).view(rows, 3)
                shards, owned = [], []
                for rank in range(size):
                    layout = CPDecodeQueryLayout(rows, rank, size)
                    shard = layout.select(source, value=-1)
                    self.assertEqual(shard.shape[0], layout.local_rows)
                    torch.testing.assert_close(
                        shard[: layout.actual_rows], source[layout.query_rows]
                    )
                    self.assertTrue((shard[layout.actual_rows :] == -1).all())
                    owned.extend(range(layout.start, layout.stop))
                    shards.append(shard)
                self.assertEqual(owned, list(range(rows)))
                torch.testing.assert_close(torch.cat(shards)[:rows], source)

    def test_only_opted_in_target_modes_use_dp(self):
        parallel = SimpleNamespace(
            cp_decode_attn_mode="dp", attn_cp_rank=2, attn_cp_size=4
        )
        with patch(
            "sglang.srt.layers.cp.cp_decode_attn_dp.get_parallel", return_value=parallel
        ):
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                batch = SimpleNamespace(forward_mode=mode)
                layout = cp_decode_query_layout(batch, 17)
                self.assertEqual((layout.start, layout.stop), (10, 15))
                self.assertIsNone(cp_decode_query_layout(batch, 17, is_draft=True))
            for mode in (
                ForwardMode.EXTEND,
                ForwardMode.DRAFT_EXTEND_V2,
                ForwardMode.IDLE,
            ):
                self.assertIsNone(
                    cp_decode_query_layout(SimpleNamespace(forward_mode=mode), 17)
                )
            parallel.cp_decode_attn_mode = None
            self.assertIsNone(
                cp_decode_query_layout(
                    SimpleNamespace(forward_mode=ForwardMode.DECODE), 17
                )
            )

    def test_indexer_page_transform_and_output_views_use_the_rank_offset(self):
        """DeepSelect must not translate every rank's picks through page rows zero."""
        backend = object.__new__(DeepseekV4AttnBackend)
        pages = torch.arange(11 * 7).view(11, 7).int()
        lens = torch.arange(11).int()
        sparse = torch.full((11, 5), -1, dtype=torch.int32)
        paged = SimpleNamespace(page_table=pages, compressed_seq_lens=lens)
        backend.forward_metadata = SimpleNamespace(
            c1_indexer_metadata=paged,
            core_metadata=SimpleNamespace(sparse_page_indices=lambda _: sparse),
        )
        layout = CPDecodeQueryLayout(11, 2, 4)
        layer = SimpleNamespace(
            compress_ratio=1, indexer=object(), layer_id=20, freqs_cis=torch.ones(1)
        )
        values = torch.arange(11)
        inputs = backend._make_low_ratio_decode_indexer_inputs(
            layer,
            values[layout.query_rows],
            values[layout.query_rows],
            values[layout.query_rows],
            values[layout.query_rows],
            ForwardMode.TARGET_VERIFY,
            query_rows=layout.query_rows,
        )
        torch.testing.assert_close(inputs.paged_metadata.page_table, pages[6:9])
        torch.testing.assert_close(inputs.paged_metadata.compressed_seq_lens, lens[6:9])
        inputs.out_page_indices.fill_(42)
        self.assertTrue((sparse[6:9] == 42).all())
        self.assertTrue((sparse[:6] == -1).all())
        self.assertTrue((sparse[9:] == -1).all())
        self.assertIs(paged.page_table, pages)
        self.assertEqual(paged.page_table.shape[0], 11)

    def test_v4_c4_scores_local_rows_but_updates_index_k_on_every_rank(self):
        """A nonzero offset uses the wrong page table if it reuses the global plan."""
        positions = torch.arange(5)
        page_table = torch.arange(5, dtype=torch.int32)[:, None]
        seq_lens = torch.tensor([1, 3, 2, 4, 1], dtype=torch.int32)
        sparse = torch.full((5, 4), -1, dtype=torch.int32)
        index_k_writes = torch.full((5,), -1)

        def score(q, cache, weights, lengths, pages, plan, width, clean_logits):
            torch.testing.assert_close(plan, lengths)
            torch.testing.assert_close(q[:, 0, 0, 0].int(), pages[:, 0])
            self.assertEqual(q.shape[0], 2)
            return torch.zeros((q.shape[0], width))

        deep_gemm = SimpleNamespace(
            get_paged_mqa_logits_metadata=lambda lengths, *_: lengths.clone(),
            get_num_sms=lambda: 1,
            fp8_paged_mqa_logits=score,
        )
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.hisparse_coordinator = None
        backend.dsa_topk_backend = DSATopKBackend.TORCH
        backend.debug_use_external_c4_sparse_indices = False
        backend.token_to_kv_pool = SimpleNamespace(
            get_index_k_with_scale_buffer=lambda **_: torch.zeros((5, 64 * 132))
        )
        backend.forward_indexer_compressor = lambda x, *_: index_k_writes.copy_(x)
        indexer = SimpleNamespace(
            layer_id=0,
            compressor=object(),
            use_fp4_indexer=False,
            compute_weights=lambda x, **_: torch.ones((x.shape[0], 1, 1)),
            compute_q=lambda q_lora, pos, w: (pos[:, None, None], w),
        )
        core = SimpleNamespace(
            positions=positions,
            c4_sparse_page_indices=sparse,
            c4_sparse_raw_indices=None,
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch(
                "sglang.srt.layers.attention.dsv4.indexer.get_global_indexer_capturer",
                return_value=None,
            ),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
        ):
            paged = PagedIndexerMetadata(
                256, 64, page_table, seq_lens, False, force_deep_gemm_metadata=True
            )
            for rank in (1, 3):
                index_k_writes.fill_(-1)
                backend.forward_metadata = DSV4Metadata(core, paged)
                backend.forward_c4_indexer(
                    positions,
                    positions,
                    indexer,
                    batch,
                    query_layout=CPDecodeQueryLayout(5, rank, 4),
                )
                torch.testing.assert_close(index_k_writes, positions)

        expected = torch.full_like(sparse, -1)
        expected[2, :2] = torch.tensor([128, 129])
        expected[3] = torch.tensor([192, 193, 194, 195])
        torch.testing.assert_close(sparse, expected)


if __name__ == "__main__":
    unittest.main()
