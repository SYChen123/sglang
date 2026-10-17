"""The runner's CP path must retain V4.1 multimodal input preparation."""

import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups import cuda_graph_hook
from sglang.srt.layers.cp import bcg
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeepseekV41CPInputs(CustomTestCase):
    def test_image_embeddings_and_remapped_ids_reach_cp_model_body(self):
        raw_ids = torch.tensor([3, MM_PAD_SHIFT_VALUE, MM_PAD_SHIFT_VALUE + 17, 31])
        embeddings = torch.arange(8).reshape(4, 2).float()
        batch = SimpleNamespace(
            input_ids=raw_ids.clone(),
            positions=torch.arange(4),
            mm_inputs=[object()],
            forward_mode=ForwardMode.EXTEND,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=32, image_token_id=7),
            vision=object(),
            _prepare_mm_embeddings=Mock(return_value=embeddings),
            get_input_embeddings=Mock(),
            model=Mock(return_value=(embeddings[1::2], embeddings[1::2])),
            pp_group=SimpleNamespace(is_last_rank=True),
            capture_aux_hidden_states=False,
            logits_processor=Mock(return_value="logits"),
            lm_head=object(),
        )
        model.prepare_language_model_inputs = MethodType(
            DeepseekV4ForCausalLM.prepare_language_model_inputs, model
        )
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))

        @contextmanager
        def shard(full_embeds, positions, forward_batch, ids):
            self.assertIs(full_embeds, embeddings)
            self.assertIs(forward_batch, batch)
            self.assertEqual(ids.tolist(), [3, 7, 7, 31])
            yield full_embeds[1::2], positions[1::2], ids[1::2]

        with (
            patch.object(eager_runner, "cp_shard_model_inputs", shard),
            patch.object(
                eager_runner,
                "cp_gather_after_forward",
                return_value=(embeddings, embeddings),
            ),
            patch.object(torch.cuda, "current_stream", return_value=None),
        ):
            result = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
        self.assertEqual(result, "logits")
        model._prepare_mm_embeddings.assert_called_once()
        model.get_input_embeddings.assert_not_called()
        self.assertEqual(model.model.call_args.args[0].tolist(), [7, 31])
        self.assertTrue(torch.equal(batch.input_ids, raw_ids))
        self.assertTrue(
            torch.equal(model.model.call_args.kwargs["input_embeds"], embeddings[1::2])
        )

    def test_text_inputs_and_supplied_embeddings_are_preserved(self):
        model = SimpleNamespace(vision=None)
        ids, embeds = torch.tensor([1, 2]), torch.zeros(2, 4)
        actual_ids, actual_embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            model, ids, SimpleNamespace(), embeds
        )
        self.assertIs(actual_ids, ids)
        self.assertIs(actual_embeds, embeds)

    def test_verify_decode_and_idle_preserve_inputs_with_retained_images(self):
        # Eager verify retains the prompt's image metadata, but has no prefill
        # lengths. These images must not be embedded again for generated tokens.
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DECODE, ForwardMode.IDLE):
            for supplied_embeds in (False, True):
                with self.subTest(mode=mode, supplied_embeds=supplied_embeds):
                    ids = torch.tensor(
                        [] if mode.is_idle() else [1, 2], dtype=torch.long
                    )
                    embeds = torch.zeros(len(ids), 4) if supplied_embeds else None
                    mm_inputs = [SimpleNamespace(mm_items=[object()])]
                    batch = SimpleNamespace(
                        forward_mode=mode,
                        mm_inputs=mm_inputs,
                        extend_prefix_lens_cpu=None,
                        extend_seq_lens_cpu=None,
                    )
                    model = SimpleNamespace(
                        vision=object(),
                        config=SimpleNamespace(vocab_size=32, image_token_id=7),
                        _prepare_mm_embeddings=Mock(),
                    )
                    actual_ids, actual_embeds = (
                        DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, batch, embeds
                        )
                    )
                    self.assertIs(actual_ids, ids)
                    self.assertIs(actual_embeds, embeds)
                    model._prepare_mm_embeddings.assert_not_called()
                    self.assertIs(batch.mm_inputs, mm_inputs)

    def test_prompt_images_still_reject_supplied_embeddings(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(forward_mode=mode, mm_inputs=[object()])
                model = SimpleNamespace(vision=object(), _prepare_mm_embeddings=Mock())
                with self.assertRaisesRegex(
                    ValueError, "input_embeds and image inputs"
                ):
                    DeepseekV4ForCausalLM.prepare_language_model_inputs(
                        model, torch.tensor([1, 2]), batch, torch.zeros(2, 4)
                    )
                model._prepare_mm_embeddings.assert_not_called()

    def test_runner_delegates_compact_tail_logits_before_generic_gather(self):
        ids, embeds = torch.arange(3), torch.ones(3, 2)
        batch = SimpleNamespace(input_ids=ids, positions=ids)
        hidden, aux = (embeds, embeds.clone()), [embeds.clone()]
        cp_logits = Mock(return_value="compact logits")
        model = SimpleNamespace(
            prepare_cp_inputs=Mock(return_value=(embeds, ids, {})),
            model=Mock(return_value=(hidden, aux)),
            pp_group=SimpleNamespace(is_last_rank=True),
            capture_aux_hidden_states=True,
            logits_from_cp_hidden_states=cp_logits,
        )
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))

        @contextmanager
        def shard(full_embeds, positions, forward_batch, model_ids):
            yield full_embeds, positions, model_ids

        with (
            patch.object(eager_runner, "cp_shard_model_inputs", shard),
            patch.object(eager_runner, "cp_gather_after_forward") as generic_gather,
        ):
            result = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
        self.assertEqual(result, "compact logits")
        cp_logits.assert_called_once_with(hidden, aux, batch)
        generic_gather.assert_not_called()


class TestDeepseekV41CPBCGInputs(CustomTestCase):
    def test_bcg_shards_language_inputs_and_refreshes_fixed_buffers(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                strategy = InterleaveCPStrategy(cp_size=4)
                buffers = bcg.PrefillCPBCGInput(
                    input_embeds=torch.zeros(16, 2),
                    positions=torch.zeros(16, dtype=torch.int64),
                    input_ids=torch.zeros(16, dtype=torch.int64),
                    rank_major_input_ids=torch.zeros(64, dtype=torch.int64),
                    num_token_non_padded=torch.zeros((), dtype=torch.int32),
                )
                pointers = [
                    buffers.input_ids.data_ptr(),
                    buffers.positions.data_ptr(),
                    buffers.num_token_non_padded.data_ptr(),
                    buffers.rank_major_input_ids.data_ptr(),
                ]
                model = SimpleNamespace(
                    logits_from_cp_hidden_states=Mock(),
                    get_input_embeddings=Mock(),
                    prepare_language_model_inputs=Mock(
                        side_effect=lambda ids, batch, embeds: (
                            ids + 100,
                            (ids + 100).float()[:, None].expand(-1, 2),
                        )
                    ),
                )
                runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))

                def prepare(batch):
                    n = batch.extend_num_tokens
                    batch.attn_cp_metadata = InterleaveContextParallelMetadata(
                        per_rank_actual_token=[4] * 4,
                        max_rank_len=[4] * 4,
                        per_rank_logical_token=[
                            n // 4 + int(r < n % 4) for r in range(4)
                        ],
                        total_seq_lens=n,
                        bs=1,
                    )

                def shard(value, batch):
                    result = value[rank::4]
                    return torch.cat(
                        [result, result.new_zeros((4 - len(result), *result.shape[1:]))]
                    )

                for capture, n in [(True, 16), (False, 13), (False, 7)]:
                    ids = torch.arange(16, dtype=torch.int64) + n * 1000
                    positions = torch.arange(16, dtype=torch.int64) + n * 10
                    batch = SimpleNamespace(
                        input_ids=ids,
                        positions=positions,
                        extend_num_tokens=n,
                        extend_seq_lens_cpu=[n],
                        forward_mode=ForwardMode.EXTEND,
                        out_cache_loc=torch.arange(16),
                        num_token_non_padded=torch.tensor(n),
                    )
                    with (
                        patch.object(bcg, "prepare_cp_forward", side_effect=prepare),
                        patch.object(bcg, "get_cp_strategy", return_value=strategy),
                        patch(
                            "sglang.srt.layers.cp.utils.get_cp_strategy",
                            return_value=strategy,
                        ),
                        patch.object(bcg, "cp_shard_hidden_states", side_effect=shard),
                        patch.object(
                            bcg,
                            "cp_split_before_forward",
                            side_effect=lambda embeds, pos, batch: (
                                shard(embeds, batch),
                                shard(pos, batch),
                            ),
                        ),
                        patch(
                            "sglang.srt.layers.cp.base.get_parallel",
                            return_value=SimpleNamespace(attn_cp_rank=rank),
                        ),
                        patch(
                            "sglang.srt.layers.cp.utils.get_parallel",
                            return_value=SimpleNamespace(attn_cp_size=4),
                        ),
                        patch(
                            "sglang.srt.layers.cp.utils.get_moe_a2a_backend",
                            return_value=SimpleNamespace(is_none=lambda: True),
                        ),
                    ):
                        buffers.prepare(
                            runner, batch, static_num_tokens=16, capture=capture
                        )
                    self.assertIs(batch.input_ids, ids)
                    self.assertIs(batch.positions, positions)
                    torch.testing.assert_close(
                        batch._cp_input_ids, shard(ids[:n] + 100, batch)
                    )
                    torch.testing.assert_close(
                        batch._cp_positions, shard(positions[:n], batch)
                    )
                    self.assertEqual(
                        int(batch.num_token_non_padded), n // 4 + int(rank < n % 4)
                    )
                    self.assertIs(
                        batch.attn_cp_metadata.moe_local_token_count,
                        buffers.num_token_non_padded,
                    )
                    self.assertEqual(batch.attn_cp_metadata.total_seq_lens, 16)
                    expected_ids = torch.zeros(16, dtype=torch.int64)
                    expected_ids[:n] = ids[:n] + 100
                    torch.testing.assert_close(
                        batch.input_ids_global, expected_ids.view(-1, 4).T.flatten()
                    )
                    self.assertEqual(
                        pointers,
                        [
                            buffers.input_ids.data_ptr(),
                            buffers.positions.data_ptr(),
                            buffers.num_token_non_padded.data_ptr(),
                            buffers.rank_major_input_ids.data_ptr(),
                        ],
                    )
                model.get_input_embeddings.assert_not_called()

    def test_cp_bcg_keeps_eager_default_and_honors_explicit_opt_in(self):
        cfg = SimpleNamespace(pp_size=1, cuda_graph_config=CudaGraphConfig())
        cfg.cuda_graph_config.prefill.backend = Backend.BREAKABLE
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", architectures=["DeepseekV4ForCausalLM"]
            )
        )
        for explicit in (False, True):
            args = SimpleNamespace(
                _cuda_graph_config_locked={("prefill", "backend")}
                if explicit
                else set()
            )
            with (
                self.subTest(explicit=explicit),
                patch.object(cuda_graph_hook, "resolving_view", return_value=cfg),
                patch.object(cuda_graph_hook, "model_config_of", return_value=model),
                patch.object(cuda_graph_hook, "declare_resolution") as declare,
            ):
                cuda_graph_hook.apply_prefill_cp_cuda_graph_default(args)
                declare.assert_not_called()
                cuda_graph_hook.apply_cuda_graph_compatibility(args)
                if explicit:
                    declare.assert_not_called()
                else:
                    self.assertEqual(
                        declare.call_args.kwargs["cuda_graph_config"].prefill.backend,
                        Backend.DISABLED,
                    )

    def test_cp_bcg_support_is_restricted_to_valid_model_and_topology(self):
        for strategy, backend, model_type, cp, tp, pp, expected in [
            ("interleave", "dsv4", "deepseek_v41", 4, 4, 1, True),
            ("interleave", "dsv4", "llama", 4, 4, 1, False),
            ("interleave", "dsv4", "deepseek_v41", 2, 4, 1, False),
            ("interleave", "dsv4", "deepseek_v41", 4, 4, 2, False),
            ("zigzag", "trtllm_mha", "llama", 4, 4, 1, True),
        ]:
            cfg = SimpleNamespace(
                enable_prefill_cp=True, cp_strategy=strategy, tp_size=tp, pp_size=pp
            )
            with (
                self.subTest(
                    strategy=strategy,
                    backend=backend,
                    model_type=model_type,
                    cp=cp,
                    tp=tp,
                    pp=pp,
                ),
                patch.object(bcg, "resolving_view", return_value=cfg),
                patch.object(
                    bcg,
                    "resolved_view",
                    return_value=SimpleNamespace(attn_cp_size=cp),
                ),
                patch.object(
                    bcg, "attention_backends_of", return_value=(backend, backend)
                ),
                patch.object(
                    bcg,
                    "model_config_of",
                    return_value=SimpleNamespace(
                        hf_config=SimpleNamespace(
                            model_type=model_type,
                            architectures=["DeepseekV4ForCausalLM"]
                            if model_type == "deepseek_v41"
                            else ["LlamaForCausalLM"],
                        )
                    ),
                ),
            ):
                self.assertEqual(bcg.supports_prefill_cp_bcg(object()), expected)


if __name__ == "__main__":
    unittest.main()
