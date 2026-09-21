"""MiniMind 法律回答解码策略的回归测试。"""

import unittest
from types import SimpleNamespace

from minimind.model.model_minimind import (
    MiniMindForCausalLM,
    _apply_generated_repetition_penalties,
    _contrastive_search_scores,
)

import torch

from minimind.trainer.chat_current_law_rag import MiniMindGenerator


class _FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, prompt, **kwargs):
        return {
            "input_ids": torch.tensor([[99]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        return " ".join(str(int(token_id)) for token_id in token_ids)


class _RepeatingModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[99, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=torch.long)


class MiniMindGeneratorDecodingTest(unittest.TestCase):
    def test_uses_repetition_controls_and_trims_repeating_tail(self):
        model = _RepeatingModel()
        generator = MiniMindGenerator(
            model=model,
            tokenizer=_FakeTokenizer(),
            device=torch.device("cpu"),
        )

        output = generator(
            [{"role": "user", "content": "测试重复输出"}],
            temperature=0,
            max_tokens=32,
        )

        self.assertEqual(1, len(model.calls))
        call = model.calls[0]
        self.assertFalse(call["do_sample"])
        self.assertGreater(call["repetition_penalty"], 1.0)
        self.assertGreaterEqual(call["no_repeat_ngram_size"], 2)
        self.assertGreater(call["repetition_control_start_tokens"], 0)
        self.assertEqual(0, call["repetition_tail_repeat_count"])
        self.assertEqual("1 2 3 1 2 3", output)


class _ControlledRepeatingModel(MiniMindForCausalLM):
    """每一步都偏好 token 1，用于稳定触发重复退化。"""

    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True, **kwargs):
        self.forward_calls += 1
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -100.0)
        logits[..., 1] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class _PromptSensitiveModel(MiniMindForCausalLM):
    """固定偏好一个 token，用于验证 prompt 不参与重复抑制。"""

    def __init__(self, *, preferred_token, fallback_token, vocab_size=128):
        self.preferred_token = preferred_token
        self.fallback_token = fallback_token
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -100.0)
        logits[..., self.preferred_token] = 10.0
        logits[..., self.fallback_token] = 9.5
        return SimpleNamespace(logits=logits, past_key_values=None)


class MiniMindGenerateRepetitionTest(unittest.TestCase):
    def test_contrastive_search_prefers_less_similar_generated_path(self):
        probabilities = torch.tensor([[0.60, 0.40]])
        candidates = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        history = torch.tensor([[[1.0, 0.0]]])

        scores = _contrastive_search_scores(
            probabilities,
            candidates,
            history,
            penalty_alpha=0.6,
        )

        self.assertEqual(1, scores.argmax(dim=-1).item())

    def test_contrastive_search_has_no_prompt_history_penalty(self):
        probabilities = torch.tensor([[0.60, 0.40]])
        candidates = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        scores = _contrastive_search_scores(
            probabilities,
            candidates,
            None,
            penalty_alpha=0.6,
        )

        self.assertEqual(0, scores.argmax(dim=-1).item())

    def test_frequency_penalty_starts_before_third_generated_occurrence(self):
        logits = torch.tensor([[0.0, 10.0, 9.0]])

        twice = _apply_generated_repetition_penalties(
            logits.clone(),
            torch.tensor([[1]]),
            frequency_penalty=0.5,
        )
        third = _apply_generated_repetition_penalties(
            logits.clone(),
            torch.tensor([[1, 1]]),
            frequency_penalty=0.5,
        )

        self.assertEqual(10.0, twice[0, 1].item())
        self.assertEqual(9.5, third[0, 1].item())
        self.assertEqual(9.0, third[0, 2].item())

    def test_repeated_path_is_softly_penalized(self):
        logits = torch.tensor([[0.0, 8.0, 7.0, 6.0, 10.0, 9.0]])

        adjusted = _apply_generated_repetition_penalties(
            logits.clone(),
            torch.tensor([[1, 2, 3, 4, 1, 2, 3]]),
            repetition_path_penalty=0.75,
            repetition_path_min_ngram_size=4,
            repetition_path_max_ngram_size=4,
        )

        self.assertEqual(9.25, adjusted[0, 4].item())
        self.assertEqual(9.0, adjusted[0, 5].item())
        self.assertTrue(torch.isfinite(adjusted).all())

    def test_frequency_penalty_ignores_prompt_occurrences(self):
        model = _PromptSensitiveModel(preferred_token=7, fallback_token=8)
        output = model.generate(
            inputs=torch.tensor([[7, 7, 7]], dtype=torch.long),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            max_new_tokens=1,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            frequency_penalty=1.0,
            repetition_path_penalty=1.0,
            eos_token_id=None,
            use_cache=False,
        )
        self.assertEqual([7], output[0, 3:].tolist())

    def test_repetition_penalty_ignores_prompt_tokens(self):
        model = _PromptSensitiveModel(preferred_token=7, fallback_token=8)
        output = model.generate(
            inputs=torch.tensor([[7]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            max_new_tokens=1,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=0,
            repetition_tail_repeat_count=0,
            eos_token_id=None,
            use_cache=False,
        )
        self.assertEqual([7], output[0, 1:].tolist())

    def test_repetition_controls_can_start_after_generated_prefix(self):
        model = _PromptSensitiveModel(preferred_token=7, fallback_token=8)
        output = model.generate(
            inputs=torch.tensor([[99]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            max_new_tokens=3,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=0,
            repetition_control_start_tokens=2,
            repetition_tail_repeat_count=0,
            eos_token_id=None,
            use_cache=False,
        )
        self.assertEqual([7, 7, 8], output[0, 1:].tolist())

    def test_no_repeat_ngram_ignores_prompt_ngrams(self):
        model = _PromptSensitiveModel(preferred_token=8, fallback_token=9)
        output = model.generate(
            inputs=torch.tensor([[5, 6, 7, 8, 5, 6, 7]], dtype=torch.long),
            attention_mask=torch.ones((1, 7), dtype=torch.long),
            max_new_tokens=1,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=4,
            repetition_tail_repeat_count=0,
            eos_token_id=None,
            use_cache=False,
        )
        self.assertEqual([8], output[0, 7:].tolist())

    def test_controlled_logits_reproduce_repeat_without_controls(self):
        model = _ControlledRepeatingModel()
        output = model.generate(
            inputs=torch.tensor([[99]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            max_new_tokens=8,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            repetition_tail_repeat_count=0,
            eos_token_id=None,
            use_cache=False,
        )
        self.assertEqual([1] * 8, output[0, 1:].tolist())

    def test_repetition_tail_early_stop_in_generation_loop(self):
        model = _ControlledRepeatingModel()
        output = model.generate(
            inputs=torch.tensor([[99]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            max_new_tokens=8,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            repetition_tail_repeat_count=3,
            eos_token_id=2,
            use_cache=False,
        )
        self.assertEqual([1, 1, 1, 1, 1, 2], output[0, 1:].tolist())
        self.assertEqual(6, model.forward_calls)

    def test_no_repeat_ngram_blocks_repeated_ngram(self):
        model = _ControlledRepeatingModel()
        output = model.generate(
            inputs=torch.tensor([[99]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            max_new_tokens=8,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=4,
            repetition_tail_repeat_count=0,
            eos_token_id=None,
            use_cache=False,
        )
        generated = output[0, 1:].tolist()
        self.assertNotEqual([1] * 8, generated)
        windows = [tuple(generated[i:i + 4]) for i in range(max(0, len(generated) - 3))]
        self.assertEqual(len(windows), len(set(windows)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
