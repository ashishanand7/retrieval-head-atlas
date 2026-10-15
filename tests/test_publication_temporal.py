import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from rha.confirmation import validate_pair
from rha.confirmation_scoring import evaluate as query_evaluate
from rha.generation import cache_seq_len
from rha.interventions import OProjInputCacher
from rha.temporal import CORRUPTIONS, SCOPES, build_examples, condition_names, output_metrics, select_groups
from rha.temporal_scoring import aligned_step, continue_branch, evaluate, patch_active, prepare_prompt
from scripts.analyze_publication_temporal import analyze, paired_analysis, summarize, validate_rows
from scripts.run_publication_confirmation import make_groups
from scripts.run_publication_temporal import assert_self_patch, check_smoke, iter_case_rows, main, verify_committed_sources

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(character) for character in text]
        return {"input_ids": torch.tensor([ids])} if return_tensors else SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "User: " + messages[0]["content"] + "\nAssistant:\n"


class TemporalPromptTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "configs/publication_temporal.json").read_text())
        self.original = json.loads((ROOT / "configs/publication_confirmation.json").read_text())
        self.tokenizer = CharacterTokenizer()

    def test_frozen_groups_preserve_original_control_draws(self):
        all_groups = make_groups(self.original, 5)
        selected = select_groups(self.protocol, self.original)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(condition_names(selected)), 44)
        for name, heads in selected.items():
            self.assertEqual(heads, all_groups[name])
        self.assertEqual(selected["random_address_L22H7_3"], [(22, 10)])
        self.assertEqual(40 * 2 * len(condition_names(selected)), 3520)

    def test_corruptions_are_paired_and_seeds_disjoint(self):
        self.protocol.update(target_tokens=512, max_len=2048)
        smoke = build_examples(self.tokenizer, self.protocol, "smoke", set())
        full = build_examples(self.tokenizer, self.protocol, "confirmation", set())
        self.assertEqual(len(smoke), 10)
        self.assertEqual(len(full), 80)
        self.assertFalse({row["example_id"] for row in smoke} & {row["example_id"] for row in full})
        self.assertEqual(smoke, build_examples(self.tokenizer, self.protocol, "smoke", set()))
        for different, same in zip(smoke[::2], smoke[1::2]):
            self.assertEqual(different["clean_prompt"], same["clean_prompt"])
            self.assertEqual(different["unrelated_decoy"], same["unrelated_decoy"])
            self.assertNotEqual(different["clean_gold"][0], different["corrupt_gold"][0])
            self.assertEqual(same["clean_gold"][0], same["corrupt_gold"][0])
            self.assertNotEqual(same["clean_gold"][1], same["corrupt_gold"][1])
            self.assertNotIn("[NEEDLE", same["clean_prompt"])

    def test_overlap_and_truncation_fail_closed(self):
        self.protocol.update(target_tokens=512, max_len=2048)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_examples(self.tokenizer, self.protocol, "smoke", {("literal", 12100000)})
        self.protocol["max_len"] = 100
        with self.assertRaisesRegex(ValueError, "truncation"):
            build_examples(self.tokenizer, self.protocol, "smoke", set())
        self.protocol["stages"]["smoke"]["seed_base"] = self.protocol["stages"]["confirmation"]["seed_base"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_examples(self.tokenizer, self.protocol, "smoke", set())

    def test_suffix_metrics_are_not_full_answer_or_extraction(self):
        answers = {"clean": "123456", "corrupt": "193845", "decoy": "987650"}
        ids = {key: self.tokenizer(value).input_ids for key, value in answers.items()}
        metrics = output_metrics("923456", self.tokenizer("923456").input_ids, answers, ids)
        self.assertEqual(metrics["exact_clean"], 0)
        self.assertEqual(metrics["aligned_suffix_exact_clean"], 1)
        prose = "The code is 123456."
        metrics = output_metrics(prose, self.tokenizer(prose).input_ids, answers, ids)
        self.assertEqual(metrics["exact_clean"], 0)
        self.assertEqual(metrics["clean_number_content"], 1)
        self.assertEqual(metrics["aligned_suffix_exact_clean"], 0)

    def test_scope_truth_table(self):
        expected = {"none": [False, False, False], "prompt_only": [True, False, False],
                    "decode_only": [False, True, True], "prompt_and_decode": [True, True, True]}
        for scope, active in expected.items():
            self.assertEqual([patch_active(scope, step) for step in range(3)], active)


class TemporalScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                             attention_dropout=0.0, eos_token_id=0)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).eval()
        from rha.modeling import inspect_model_layout
        cls.layout = inspect_model_layout(cls.model)
        cls.tokenizer = CharacterTokenizer()
        cls.answers = {"clean": "123456", "corrupt": "678912", "decoy": "987650"}
        cls.clean_prompt = "Code 123456. Ignore 987650. Reply:"
        cls.corrupt_prompt = cls.clean_prompt.replace("123456", "678912")
        cls.clean = prepare_prompt(cls.model, cls.tokenizer, cls.clean_prompt, cls.answers, 128, 7, 24)
        cls.corrupt = prepare_prompt(cls.model, cls.tokenizer, cls.corrupt_prompt, cls.answers, 128, 7, 24)
        cls.pairs = [(0, 1), (1, 2)]

    def run_condition(self, scope, recipient=None, donor=None):
        return evaluate(self.model, self.tokenizer, self.layout, recipient or self.corrupt, 7,
                        donor=donor or self.clean, pairs=self.pairs, scope=scope)

    def test_self_patch_all_scopes_and_hooks_cleanup(self):
        for prepared in (self.clean, self.corrupt):
            baseline = evaluate(self.model, self.tokenizer, self.layout, prepared, 7)
            for scope in SCOPES:
                values = self.run_condition(scope, prepared, prepared)
                assert_self_patch(values, baseline)
                for event in values["generation_steps"]:
                    self.assertEqual(event["patch_applied"], patch_active(scope, event["step"]))
                    self.assertEqual(event["verified_hook_calls"], {"0": int(event["patch_applied"]), "1": int(event["patch_applied"])})
        self.assertTrue(all(not module._forward_pre_hooks for module in self.layout.o_projs))

    def test_prompt_only_matches_existing_query_runner(self):
        args = (self.model, self.tokenizer, self.layout)
        _, cache = query_evaluate(*args, self.clean_prompt, "123456", "678912", 128, 7, 7, capture=True)
        old, _ = query_evaluate(*args, self.corrupt_prompt, "123456", "678912", 128, 7, 7, pairs=self.pairs, donor=cache)
        new = self.run_condition("prompt_only")
        for key in ("clean_mean_logprob", "corrupt_mean_logprob", "answer_logprob_margin", "first_clean_logprob"):
            self.assertAlmostEqual(old[key], new[key], places=6)
        self.assertEqual(old["generated_token_ids"], new["generated_token_ids"])

    def full_prefix_reference(self, prefix, scope):
        recipient_ids = self.tokenizer(self.corrupt_prompt).input_ids + prefix
        donor_ids = self.tokenizer(self.clean_prompt).input_ids + prefix
        with torch.no_grad(), OProjInputCacher(self.layout) as captured:
            self.model(input_ids=torch.tensor([donor_ids]), use_cache=False)
        handles = []
        for layer in sorted({layer for layer, _ in self.pairs}):
            heads = [head for candidate_layer, head in self.pairs if candidate_layer == layer]

            def hook(_module, inputs, layer=layer, heads=heads):
                value = inputs[0].clone()
                view = value.reshape(1, value.shape[1], 4, 8)
                source = captured.cache[layer].reshape_as(view)
                for step in range(len(prefix) + 1):
                    if patch_active(scope, step):
                        for head in heads:
                            view[:, self.corrupt.length - 1 + step, head, :] = source[:, self.clean.length - 1 + step, head, :]
                return (value,) + tuple(inputs[1:])

            handles.append(self.layout.o_projs[layer].register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                return self.model(input_ids=torch.tensor([recipient_ids]), use_cache=False).logits[0, -1].float().log_softmax(-1)
        finally:
            for handle in handles:
                handle.remove()

    def test_all_scopes_match_independent_full_prefix_recomputation(self):
        for scope in (*SCOPES, "none"):
            values = self.run_condition(scope) if scope != "none" else evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7)
            for label in ("clean", "corrupt", "decoy"):
                tokens = self.clean.candidate_ids[label]
                for index, token in enumerate(tokens):
                    expected = float(self.full_prefix_reference(tokens[:index], scope)[token])
                    self.assertAlmostEqual(expected, values["candidate_scores"][label]["token_logprobs"][index], places=5)

    def test_decode_only_cannot_change_first_step(self):
        baseline = evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7)
        later = self.run_condition("decode_only")
        for label in self.answers:
            self.assertEqual(baseline["candidate_scores"][label]["token_logprobs"][0],
                             later["candidate_scores"][label]["token_logprobs"][0])
        self.assertEqual(baseline["generated_token_ids"][0], later["generated_token_ids"][0])

    def test_greedy_donor_receives_actual_recipient_prefix_and_stops_at_eos(self):
        import rha.temporal_scoring as scoring
        original = scoring.one_step_logits
        expected = self.tokenizer("765432").input_ids + [0]
        calls = []

        def observed(model, token, past):
            calls.append((int(token.item()), cache_seq_len(past)))
            logits, cache = original(model, token, past)
            logits = logits.clone().fill_(-1000)
            # Deliberately disagree with the donor's argmax; only recipient choices may feed back.
            chosen = expected[len(calls) // 2 - 1] if len(calls) % 2 == 0 else ord("1")
            logits[0, chosen] = 0
            return logits, cache

        with patch.object(scoring, "one_step_logits", side_effect=observed):
            start = aligned_step(self.model, self.layout, self.corrupt.last_token,
                                 copy.deepcopy(self.corrupt.past), copy.deepcopy(self.clean.past),
                                 self.pairs, "prompt_and_decode", 0, self.corrupt.length)
            result = continue_branch(self.model, self.tokenizer, self.layout, start, self.pairs,
                                     "prompt_and_decode", self.corrupt.length, 24)
        self.assertEqual(result["token_ids"], expected)
        self.assertEqual(result["stop"], "eos")
        self.assertEqual(calls[::2], calls[1::2])
        self.assertEqual([token for token, _ in calls[::2]], [int(self.corrupt.last_token.item())] + expected[:-1])
        self.assertEqual(cache_seq_len(start[1]), self.corrupt.length)
        self.assertEqual(cache_seq_len(start[2]), self.clean.length)

    def test_branch_order_and_conditions_do_not_mutate_shared_caches(self):
        before = [[tuple(tensor.clone() for tensor in layer) for layer in prepared.past.to_legacy_cache()]
                  for prepared in (self.clean, self.corrupt)]
        first = self.run_condition("prompt_and_decode")
        reversed_prepared = copy.copy(self.corrupt)
        reversed_prepared.candidate_ids = dict(reversed(list(self.corrupt.candidate_ids.items())))
        reordered = self.run_condition("prompt_and_decode", recipient=reversed_prepared)
        self.assertEqual(first, reordered)
        self.run_condition("decode_only")
        self.assertEqual(first, self.run_condition("prompt_and_decode"))
        for saved, prepared in zip(before, (self.clean, self.corrupt)):
            for expected, actual in zip(saved, prepared.past.to_legacy_cache()):
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_alignment_budget_and_hook_failures_are_closed(self):
        bad = copy.copy(self.clean)
        bad.last_token = torch.tensor([[1]])
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.run_condition("prompt_only", donor=bad)
        with self.assertRaisesRegex(ValueError, "truncated"):
            prepare_prompt(self.model, self.tokenizer, self.clean_prompt, self.answers, 2, 7, 24)
        with self.assertRaisesRegex(ValueError, "position"):
            prepare_prompt(self.model, self.tokenizer, self.clean_prompt, self.answers, 128, 7, 2048)
        with self.assertRaisesRegex(ValueError, "scope"):
            self.run_condition("all")
        import rha.temporal_scoring as scoring
        original = scoring.one_step_logits
        calls = 0

        def fail_recipient(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Injected recipient failure")
            return original(*args)

        with patch.object(scoring, "one_step_logits", side_effect=fail_recipient):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                self.run_condition("prompt_and_decode")
        self.assertTrue(all(not module._forward_pre_hooks for module in self.layout.o_projs))


class TemporalOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TemporalScoringTests.setUpClass()
        base = TemporalScoringTests
        cls.protocol = {"max_len": 128, "max_new_tokens": 3, "focal_groups": ["tiny"]}
        cls.groups = {"tiny": [(0, 1)], "random_tiny_0": [(0, 2)]}
        cls.examples, cls.rows = [], []
        for mode, other in (("different_first", "678912"), ("same_first", "193845")):
            clean, corrupt = base.clean_prompt, base.clean_prompt.replace("123456", other)
            answers = {"clean": "123456", "corrupt": other, "decoy": "987650"}
            ex = {"example_id": "literal:12100000", "case_id": f"literal:12100000/{mode}", "variant": "literal", "seed": 12100000,
                  "corruption": mode, "shared_prefix_tokens": CORRUPTIONS.index(mode), "filler_words": 0,
                  "clean_gold": "123456", "corrupt_gold": other, "unrelated_decoy": "987650",
                  "clean_prompt": clean, "corrupt_prompt": corrupt,
                  "clean_prompt_sha256": hashlib.sha256(clean.encode()).hexdigest(),
                  "corrupt_prompt_sha256": hashlib.sha256(corrupt.encode()).hexdigest(),
                  "candidate_token_ids": {key: base.tokenizer(value).input_ids for key, value in answers.items()},
                  "final_prompt_token_id": ord(":"), **validate_pair(base.tokenizer, clean, corrupt, "123456", other, 128)}
            cls.examples.append(ex)
            cls.rows.extend(iter_case_rows(base.model, base.tokenizer, base.layout, ex, cls.groups, cls.protocol, 7))
        cls.manifest = {"groups": cls.groups, "protocol": cls.protocol, "eos_token_ids": [0],
                        "rows_written": len(cls.rows), "expected_rows": 28, "n_independent_example_ids": 1, "n_example_pairs": 2}

    def test_complete_tiny_model_pipeline_and_identity_level_intervals(self):
        result = validate_rows(self.manifest, self.examples, self.rows)
        self.assertEqual(result["rows"], 28)
        self.assertEqual(result["independent_identities"], 1)
        self.assertEqual(result["self_patch_max_token_score_error"], 0)
        summary = summarize(self.rows)
        self.assertTrue(summary["by_condition"])
        paired = paired_analysis(self.manifest, self.rows, draws=30)
        self.assertTrue(all(row["n_identities"] == 1 for row in paired["contrasts"]))

    def test_missing_rows_bad_prefix_wrong_hooks_and_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            validate_rows(self.manifest, self.examples, self.rows[:-1])
        for field in ("donor_input_token_id", "donor_cache_length_before", "verified_hook_calls"):
            rows = copy.deepcopy(self.rows)
            event = next(row for row in rows if row["condition"] == "patch:prompt_and_decode:tiny")["candidate_scores"]["clean"]["steps"][1]
            event[field] = {} if field == "verified_hook_calls" else -1
            with self.assertRaisesRegex(ValueError, "trace"):
                validate_rows(self.manifest, self.examples, rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["candidate_scores"]["clean"]["sum_logprob"] += 1
        with self.assertRaisesRegex(ValueError, "token sum"):
            validate_rows(self.manifest, self.examples, rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["candidate_scores"]["clean"]["token_logprobs"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_rows(self.manifest, self.examples, rows)

    def test_self_patch_error_includes_generated_token_scores(self):
        rows = copy.deepcopy(self.rows)
        row = next(row for row in rows if row["condition"] == "self_patch_clean:prompt_only")
        row["generated_token_logprobs"][0] += 5e-6
        result = validate_rows(self.manifest, self.examples, rows)
        self.assertAlmostEqual(result["self_patch_max_token_score_error"], 5e-6, places=10)

    def test_smoke_gate_rejects_stale_and_non_smoke_runs(self):
        from scripts.run_publication_temporal import check_smoke
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = {"arguments": {"stage": "smoke", "chunk_size": 1024}, "protocol": {"revision": "abc"},
                        "source_sha256": {"file": "hash"}, "packages": {"torch": "version"},
                        "model_revision": "abc", "rows_sha256": "rows"}
            (path / "manifest.json").write_text(json.dumps(manifest))
            with patch("scripts.analyze_publication_temporal.analyze", return_value={"validation": {"status": "passed"}}):
                args = (path, {"revision": "abc"}, {"file": "hash"}, {"torch": "version"}, 1024)
                self.assertEqual(check_smoke(*args)["rows_sha256"], "rows")
                with self.assertRaisesRegex(ValueError, "stale"):
                    check_smoke(path, {"revision": "abc"}, {"file": "different"}, {"torch": "version"}, 1024)
                manifest["arguments"]["stage"] = "confirmation"
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "smoke-stage"):
                    check_smoke(*args)

    def test_partial_outputs_overwrites_and_uncommitted_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                analyze(path)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                main(["--prepare-only", "--out-dir", str(path)])
        with self.assertRaisesRegex(ValueError, "requires --smoke-dir"):
            main(["--stage", "confirmation", "--out-dir", "logs/never_created_temporal_test"])
        with self.assertRaisesRegex(ValueError, "Uncommitted"):
            verify_committed_sources({"configs/publication_confirmation.json": "wrong-hash"}, "HEAD")


if __name__ == "__main__":
    unittest.main()
