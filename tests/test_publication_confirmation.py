import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from rha.confirmation import build_examples, generation_metrics, render_prompt, validate_pair
from rha.confirmation_scoring import checked_intervention, evaluate, score_continuation
from rha.generation import cache_seq_len, one_step_logits, prefill_build_kv
from rha.modeling import inspect_model_layout, use_attn_impl
from scripts.analyze_publication_confirmation import analyze, paired_interval
from scripts.audit_publication_evidence import audit
from scripts.run_publication_confirmation import make_groups

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(character) for character in text]
        if return_tensors:
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "User: " + messages[0]["content"] + "\nAssistant:\n"


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_paired_markers_preserve_content_and_identities(self):
        rows = build_examples(self.tokenizer, 1, 512, 0.5, 9100000, "raw", 1024, set())
        self.assertEqual(len(rows), 10)
        for marked, unmarked in zip(rows[::2], rows[1::2]):
            self.assertEqual(marked["example_id"], unmarked["example_id"])
            self.assertEqual(marked["corrupt_gold"], unmarked["corrupt_gold"])
            self.assertEqual(render_prompt(self.tokenizer, marked["clean_prompt"], "unmarked", "raw"), unmarked["clean_prompt"])
            self.assertNotIn("[NEEDLE", unmarked["clean_prompt"])
            self.assertLess(unmarked["prompt_tokens"], marked["prompt_tokens"])
        self.assertEqual(rows, build_examples(self.tokenizer, 1, 512, 0.5, 9100000, "raw", 1024, set()))

    def test_discovery_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_examples(self.tokenizer, 1, 512, 0.1, 1234, "raw", 1024, {("literal", 1234)})

    def test_pair_validation_rejects_truncation_and_other_changes(self):
        with self.assertRaisesRegex(ValueError, "truncation"):
            validate_pair(self.tokenizer, "Code 123456.", "Code 654321.", "123456", "654321", 5)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_pair(self.tokenizer, "Code 123456.", "Pass 654321.", "123456", "654321", 100)

    def test_builder_rejects_overlength_before_corruption_retries(self):
        with self.assertRaisesRegex(ValueError, "truncation"):
            build_examples(self.tokenizer, 1, 512, 0.1, 9100000, "chat", 100, set())

    def test_exact_match_is_not_answer_extraction(self):
        metrics = generation_metrics("The code is 123456.", "123456", "654321")
        self.assertEqual(metrics["exact_clean"], 0)
        self.assertEqual(metrics["sole_number_is_clean"], 1)
        self.assertEqual(generation_metrics(" 123456\n", "123456", "654321")["exact_clean"], 1)
        self.assertEqual(generation_metrics("123456 or 654321", "123456", "654321")["sole_number_is_clean"], 0)

    def test_random_controls_match_layers_and_counts(self):
        protocol = json.loads((ROOT / "configs/publication_confirmation.json").read_text())
        groups = make_groups(protocol, 3)
        self.assertEqual(groups, make_groups(protocol, 3))
        for name, pairs in protocol["groups"].items():
            for draw in range(3):
                control = groups[f"random_{name}_{draw}"]
                self.assertEqual(Counter(layer for layer, _ in pairs), Counter(layer for layer, _ in control))
                self.assertFalse(set(map(tuple, pairs)) & set(control))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=256,
                             attention_dropout=0.0, eos_token_id=0)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).eval()
        cls.layout = inspect_model_layout(cls.model)
        cls.tokenizer = CharacterTokenizer()

    def test_chunked_prefill_matches_full_forward(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = "Read this code: 123456. Answer:"
        with torch.no_grad(), use_attn_impl(model, "sdpa"):
            full = model(**tokenizer(prompt, return_tensors="pt"), use_cache=False).logits[:, -1, :]
            for size in (1, 5, 128):
                past, ids = prefill_build_kv(model, tokenizer, prompt, 128, size)
                logits, _ = one_step_logits(model, ids[:, -1:], past)
                torch.testing.assert_close(logits, full, atol=1e-6, rtol=1e-5)

    def test_scoring_matches_full_teacher_forcing_and_does_not_mutate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt, answer = "Code:", "123456"
        with torch.no_grad(), use_attn_impl(model, "sdpa"):
            past, ids = prefill_build_kv(model, tokenizer, prompt, 128, 3)
            logits, past = one_step_logits(model, ids[:, -1:], past)
            before = cache_seq_len(past)
            score = score_continuation(model, logits, past, tokenizer(answer).input_ids)
            self.assertEqual(cache_seq_len(past), before)
            all_ids = tokenizer(prompt + answer, return_tensors="pt")["input_ids"]
            out = model(input_ids=all_ids, use_cache=False)
            aligned = out.logits[:, len(prompt) - 1:-1, :].float().log_softmax(-1)
            expected = aligned.gather(-1, all_ids[:, len(prompt):].unsqueeze(-1)).mean().item()
            self.assertAlmostEqual(score["mean_logprob"], expected, places=6)

    def test_self_patch_and_query_only_scope(self):
        args = (self.model, self.tokenizer, self.layout, "Code: 123456. Reply:", "123456", "654321", 128, 5, 4)
        baseline, cache = evaluate(*args, capture=True)
        patched, _ = evaluate(*args, pairs=[(0, 1), (1, 2)], donor=cache)
        self.assertEqual(baseline, patched)
        ablated, _ = evaluate(*args, pairs=[(1, 2)])
        self.assertNotEqual(baseline["clean_mean_logprob"], ablated["clean_mean_logprob"])
        self.assertTrue(all(not module._forward_pre_hooks for module in self.layout.o_projs))

    def test_invalid_donor_and_unfired_hook_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "donor"):
            with checked_intervention(self.layout, [(0, 1)], {}):
                pass
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            with checked_intervention(self.layout, [(0, 1)]):
                pass
        self.assertTrue(all(not module._forward_pre_hooks for module in self.layout.o_projs))

    def test_evaluation_never_truncates(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            evaluate(self.model, self.tokenizer, self.layout, "Long prompt", "123456", "654321", 2, 2, 2)


class EvidenceTests(unittest.TestCase):
    def test_historical_numbers_and_shared_examples(self):
        result = audit()
        self.assertEqual(result["unique_example_identities_across_settings"], 40)
        self.assertEqual(result["identities_shared_by_all_settings"], 40)
        self.assertTrue(all(row["overlap_with_discovery"] == 40 for row in result["settings"]))
        self.assertAlmostEqual(result["settings"][0]["l22h7_patch"], 0.4571, places=4)
        self.assertGreater(result["interaction_control"]["non_address_core"]["patch_effect_under_ablation"], 0.4)

    def test_bootstrap_preserves_constant_paired_effect(self):
        result = paired_interval([2.0] * 6, ["a"] * 3 + ["b"] * 3, draws=200)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["ci_low"], 2.0)
        self.assertEqual(result["ci_high"], 2.0)

    def test_partial_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                analyze(path, 100)

    def test_analysis_complete_grid_and_corrupt_grid(self):
        protocol = json.loads((ROOT / "configs/publication_confirmation.json").read_text())
        groups = make_groups(protocol, 1)
        conditions = ["clean", "corrupt", "self_patch_clean", "self_patch_corrupt"]
        conditions += [f"{action}:{name}" for name in groups for action in ("ablate", "patch")]
        rows = []
        for identity, variant in (("a:1", "a"), ("b:2", "b")):
            for mode in ("marked", "unmarked"):
                for condition in conditions:
                    delta = 0.25 if condition == "patch:address_L22H7" else 0.0
                    rows.append({"example_id": identity, "variant": variant, "marker_mode": mode,
                                 "condition": condition, "delta_clean_logprob": delta,
                                 "delta_answer_margin": delta, "delta_exact_clean": 0,
                                 "exact_clean": 0, "clean_mean_logprob": -2.0 + delta})
        manifest = {"status": "complete", "groups": groups, "protocol": protocol,
                    "arguments": {"stage": "pilot"}, "n_independent_example_ids": 2,
                    "rows_written": len(rows)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = analyze(path, 50)
            primary = next(row for row in result["contrasts"] if row["contrast"] == "marked:patch_L22H7_minus_L20H7" and row["metric"] == "delta_clean_logprob")
            self.assertEqual(primary["mean"], 0.25)
            self.assertEqual(primary["n_identities"], 2)
            (path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
            with self.assertRaisesRegex(ValueError, "grid"):
                analyze(path, 50)


if __name__ == "__main__":
    unittest.main()
