"""Cross-node placement and scoped logits layout regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.engine import _calculate_rank_ranges
from sglang.srt.layers.logits_processor import (
    _reassemble_tp_lm_head_all_to_all_output,
    compute_dp_tp_lm_head,
)
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestParallelConsumers(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _publish(self, **config):
        publish(
            ServerArgs(model_path="dummy", **config),
            role="scheduler",
            ranks=SpawnRanks(world_rank=0),
        )

    def test_rank_ranges_can_describe_another_node(self):
        self._publish(tp_size=4, pp_size=2, nnodes=4, node_rank=0)
        expected = [
            ([0], [0, 1]),
            ([0], [2, 3]),
            ([1], [0, 1]),
            ([1], [2, 3]),
        ]
        for node_rank, (pp, tp) in enumerate(expected):
            with self.subTest(node_rank=node_rank):
                pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
                    node_rank
                )
                self.assertEqual((list(pp_range), list(tp_range)), (pp, tp))
                self.assertEqual((pp_per_node, tp_per_node), (1, 2))

    def test_logits_reassembly_uses_the_active_tp_scope(self):
        self._publish(tp_size=4)
        source = torch.arange(24).reshape(8, 3)
        with get_parallel().override(tp_size=2, attn_tp_size=2, moe_tp_size=2):
            result = _reassemble_tp_lm_head_all_to_all_output(source)
            torch.testing.assert_close(result, torch.cat(source.chunk(2), dim=1))
        result = _reassemble_tp_lm_head_all_to_all_output(source)
        torch.testing.assert_close(result, torch.cat(source.chunk(4), dim=1))

    def test_dp_tp_projection_matches_full_weights_with_idle_and_padding(self):
        """Vocabulary/row exchange must preserve full logits, including idle ranks."""
        torch.manual_seed(19)
        world, hidden, vocab, shard = 4, 8, 17, 5
        self._publish(tp_size=world, attn_dp_size=world)
        weight = torch.randn(world * shard, hidden)
        for counts, graph in (
            ([10] * 4, True),
            ([10, 5, 0, 15], False),
            ([0] * 4, False),
        ):
            capacity = max(counts)
            rows = [torch.randn(n, hidden) for n in counts]
            padded = [
                torch.nn.functional.pad(x, (0, 0, 0, capacity - n))
                for x, n in zip(rows, counts)
            ]
            gathered = torch.cat(padded)
            for rank in range(world):
                with self.subTest(counts=counts, rank=rank):
                    local_weight = weight[rank * shard : (rank + 1) * shard]

                    class Group:
                        world_size = world

                        def all_gather_into_tensor(self, output, local):
                            torch.testing.assert_close(local, padded[rank])
                            output.copy_(gathered)

                        def all_to_all_single(self, output, local):
                            # Reference the complete matrix, then form the wire
                            # contributions independently for this destination.
                            torch.testing.assert_close(
                                local.view(world * capacity, shard),
                                torch.nn.functional.linear(gathered, local_weight),
                            )
                            full = torch.nn.functional.linear(padded[rank], weight)
                            output.copy_(torch.cat(full.split(shard, dim=1)).flatten())

                    head = SimpleNamespace(tp_size=world, org_vocab_size=vocab)
                    with get_parallel().override(
                        tp_rank=rank, attn_dp_rank=rank, tp_group=Group()
                    ):
                        logits = compute_dp_tp_lm_head(
                            rows[rank],
                            head,
                            lambda x: torch.nn.functional.linear(x, local_weight),
                            dp_num_rows=None if graph else counts,
                        )
                    expected = torch.nn.functional.linear(rows[rank], weight[:vocab])
                    torch.testing.assert_close(logits, expected)

    def test_dspark_all_to_all_admits_v4_and_rejects_other_drafts(self):
        """DSpark may use a TP head without silently admitting unsupported models."""
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.arg_groups.speculative_hook import _handle_dspark

        for arch in ("DeepseekV4ForCausalLM", "LlamaForCausalLM"):
            args = ServerArgs(
                model_path="dummy",
                device="cuda",
                tp_size=4,
                attn_dp_size=4,
                enable_tp_lm_head_all_to_all=True,
                enable_dp_lm_head=False,
                speculative_algorithm="DSPARK",
                speculative_draft_model_path="dummy",
                speculative_dspark_block_size=5,
                moe_a2a_backend="none",
            )
            config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
            with (
                patch(
                    "sglang.srt.arg_groups.speculative_hook.model_config_of",
                    return_value=config,
                ),
                patch(
                    "sglang.srt.speculative.dspark_components.dspark_config.read_draft_checkpoint_config",
                    return_value=None,
                ),
            ):
                if arch == "LlamaForCausalLM":
                    with self.assertRaisesRegex(ValueError, "DeepSeek-V4"):
                        _handle_dspark(args)
                else:
                    _handle_dspark(args)
                    self.assertEqual(
                        resolution_result(args, "speculative_num_draft_tokens"), 6
                    )


if __name__ == "__main__":
    unittest.main()
