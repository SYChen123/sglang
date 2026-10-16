"""The runner's CP path must retain V4.1 multimodal input preparation."""

import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
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


if __name__ == "__main__":
    unittest.main()
