import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from rha.generation import cache_seq_len
from rha.interventions import OProjInputCacher
from rha.modeling import inspect_model_layout
from rha.specificity import (ABLATION_VIEWS, VIEW_CORRUPTIONS, boundary_metrics, build_examples,
                             condition_specs, reserved_ids, select_groups)
from rha.specificity_scoring import (capture_reference, continue_branch, evaluate, final_residual,
                                     intervention_step, prepare_prompt)
from rha.temporal_scoring import evaluate as temporal_evaluate
from scripts.analyze_publication_specificity import analyze, paired_analysis, summarize, validate_design, validate_rows
from scripts.run_publication_confirmation import make_groups
from scripts.run_publication_specificity import check_smoke, iter_case_rows, main
from scripts.run_publication_temporal import assert_self_patch
from test_publication_temporal import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]


def tiny_protocol():
    protocol = json.loads((ROOT / "configs/publication_specificity.json").read_text())
    protocol.update(target_tokens=512, max_len=2048, max_new_tokens=3)
    return protocol


class SpecificityPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = tiny_protocol()
        cls.tokenizer = CharacterTokenizer()
        cls.examples, cls.references = build_examples(cls.tokenizer, cls.protocol, "smoke", set())

    def test_fixed_controls_and_nonfactorial_grid(self):
        original = json.loads((ROOT / "configs/publication_confirmation.json").read_text())
        groups = select_groups(self.protocol, original)
        old = make_groups(original, 5)
        self.assertEqual(len(groups), 18)
        self.assertTrue(all(pairs == old[name] for name, pairs in groups.items()))
        self.assertEqual(len(self.examples), 40)
        self.assertEqual(len(self.references), 25)
        self.assertEqual(sum(len(condition_specs(e, groups, self.protocol)) for e in self.examples), 1580)
        self.assertEqual(1580 * 8, 12640)

    def test_pairing_positions_references_and_boundaries(self):
        self.assertEqual((self.examples, self.references), build_examples(self.tokenizer, self.protocol, "smoke", set()))
        by_key = {(e["example_id"], e["view"], e["corruption"]): e for e in self.examples}
        for identity in {e["example_id"] for e in self.examples}:
            far = by_key[identity, "far_chat", "different_first"]
            near = by_key[identity, "near_chat", "different_first"]
            short = by_key[identity, "short_chat", "different_first"]
            raw = by_key[identity, "far_raw", "different_first"]
            space = by_key[identity, "far_raw_space", "different_first"]
            self.assertEqual(far["prompt_tokens"], near["prompt_tokens"])
            self.assertGreater(far["source_to_question_tokens"], near["source_to_question_tokens"])
            self.assertEqual(near["source_to_question_tokens"], short["source_to_question_tokens"])
            self.assertEqual(space["clean_prompt"], raw["clean_prompt"] + " ")
            self.assertEqual(space["corrupt_prompt"], raw["corrupt_prompt"] + " ")
            self.assertEqual(space["corrupt_gold"], far["corrupt_gold"])
            self.assertNotIn("[NEEDLE", far["clean_prompt"])
            self.assertLess(near["clean_prompt"].index(near["clean_gold"]), near["clean_prompt"].index("Distractor note:"))
        for ref in self.references:
            self.assertIn(ref["view"], ABLATION_VIEWS)
            self.assertEqual(ref["seed"], int(ref["paired_example_id"].split(":")[1]) + 50000)

    def test_seed_separation_and_truncation_fail_closed(self):
        ids = [reserved_ids(self.protocol, stage, ref) for stage in ("smoke", "confirmation") for ref in (False, True)]
        self.assertEqual(sum(map(len, ids)), len(set.union(*ids)))
        for reserved in (ids[0], ids[1]):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                build_examples(self.tokenizer, self.protocol, "smoke", reserved)
        bad = copy.deepcopy(self.protocol)
        bad["stages"]["confirmation"] = bad["stages"]["smoke"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_examples(self.tokenizer, bad, "smoke", set())
        bad = dict(self.protocol, max_len=30)
        with self.assertRaisesRegex(ValueError, "truncation"):
            build_examples(self.tokenizer, bad, "smoke", set())

    def test_verbatim_and_resampled_content_are_distinct(self):
        metrics = boundary_metrics(" 123456", "123456", "987654")
        self.assertEqual(metrics["verbatim_clean"], 0)
        self.assertEqual(metrics["generated_leading_whitespace"], 1)
        self.assertEqual(boundary_metrics("987654 and 987654", "123456", "987654")["resample_number_content"], 1)
        self.assertEqual(boundary_metrics("123456 and 987654", "123456", "987654")["resample_number_content"], 0)


class TinyFixture:
    @classmethod
    def initialize(cls):
        torch.set_num_threads(1)
        torch.manual_seed(516)
        cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         attention_dropout=0., eos_token_id=0)
        cfg._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.layout, cls.tokenizer = inspect_model_layout(cls.model), CharacterTokenizer()
        cls.answers = {"clean": "123456", "corrupt": "678912", "decoy": "987650"}
        cls.clean_text = "Code 123456. Ignore 987650. Reply:"
        cls.corrupt_text = cls.clean_text.replace("123456", "678912")
        cls.clean = prepare_prompt(cls.model, cls.tokenizer, cls.clean_text, cls.answers, 128, 7, 24)
        cls.corrupt = prepare_prompt(cls.model, cls.tokenizer, cls.corrupt_text, cls.answers, 128, 7, 24)
        cls.pairs = [(0, 1), (1, 2)]
        cls.reference_text = "A different code is 345678. Reply:"
        cls.reference = {"reference_id": "test-reference", "prompt": cls.reference_text,
                         "prompt_tokens": len(cls.reference_text)}
        cls.resample = capture_reference(cls.model, cls.tokenizer, cls.layout, cls.reference, 128, 7)


class SpecificityScoringTests(unittest.TestCase, TinyFixture):
    @classmethod
    def setUpClass(cls):
        cls.initialize()

    def test_delegated_head_patches_match_frozen_temporal_runner(self):
        for scope in ("prompt_only", "prompt_and_decode"):
            old = temporal_evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, self.clean, self.pairs, scope)
            new = evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, "patch", self.pairs, scope, self.clean)
            self.assertEqual(old, new)

    def test_positive_matches_donor_not_known_gold_and_all_logits(self):
        baseline = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7)
        positive = evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, "residual", scope="prompt_and_decode", donor=self.clean)
        assert_self_patch(positive, baseline)
        self.assertNotEqual(baseline["generated_text"].strip(), self.answers["clean"])
        events = positive["generation_steps"] + [e for s in positive["candidate_scores"].values() for e in s["steps"]]
        self.assertTrue(all(e["donor_logit_max_abs_error"] == 0 for e in events))
        self.assertTrue(all(e["verified_final_norm_calls"] == {"donor": 1, "recipient": 1} for e in events))

    def test_positive_control_in_cpu_bfloat16(self):
        model = copy.deepcopy(self.model).to(dtype=torch.bfloat16)
        layout = inspect_model_layout(model)
        clean = prepare_prompt(model, self.tokenizer, self.clean_text, self.answers, 128, 7, 7)
        corrupt = prepare_prompt(model, self.tokenizer, self.corrupt_text, self.answers, 128, 7, 7)
        baseline = evaluate(model, self.tokenizer, layout, clean, 7)
        positive = evaluate(model, self.tokenizer, layout, corrupt, 7, "residual", scope="prompt_and_decode", donor=clean)
        assert_self_patch(positive, baseline)
        self.assertTrue(all(e["donor_logit_max_abs_error"] == 0 for e in positive["generation_steps"]))

    def full_prefix_reference(self, prefix, action):
        ids = self.tokenizer(self.clean_text).input_ids + prefix
        handles = []
        if action == "resample":
            with torch.no_grad(), OProjInputCacher(self.layout) as captured:
                self.model(input_ids=torch.tensor([self.tokenizer(self.reference_text).input_ids]), use_cache=False)
            source = captured.cache
        for layer in sorted({layer for layer, _ in self.pairs}):
            heads = [head for selected, head in self.pairs if selected == layer]

            def hook(_module, inputs, layer=layer, heads=heads):
                value = inputs[0].clone()
                view = value.reshape(1, value.shape[1], 4, 8)
                for head in heads:
                    view[:, self.clean.length - 1, head] = 0 if action == "zero" else source[layer].reshape(1, -1, 4, 8)[:, -1, head]
                return (value,) + tuple(inputs[1:])

            handles.append(self.layout.o_projs[layer].register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                return self.model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float().log_softmax(-1)
        finally:
            for handle in handles:
                handle.remove()

    def test_zero_and_resample_match_independent_full_prefix_forward(self):
        for action in ("zero", "resample"):
            values = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7, action, self.pairs,
                              "prompt_only", resample=self.resample if action == "resample" else None)
            for label, ids in self.clean.candidate_ids.items():
                for i, token in enumerate(ids):
                    expected = float(self.full_prefix_reference(ids[:i], action)[token])
                    self.assertAlmostEqual(expected, values["candidate_scores"][label]["token_logprobs"][i], places=5)
            self.assertTrue(all(not e["patch_applied"] for e in values["generation_steps"][1:]))
            if action == "resample":
                self.assertEqual(values["generation_steps"][0]["resample_query_position"], len(self.reference_text) - 1)
                self.assertIsNone(values["generation_steps"][0]["donor_input_token_id"])

    def test_resample_self_and_branch_order_preserve_caches(self):
        reference = {"reference_id": "self", "prompt": self.clean_text, "prompt_tokens": self.clean.length}
        source = capture_reference(self.model, self.tokenizer, self.layout, reference, 128, 7)
        before = [tuple(t.clone() for t in layer) for layer in self.clean.past.to_legacy_cache()]
        baseline = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7)
        resampled = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7, "resample", self.pairs, "prompt_only", resample=source)
        assert_self_patch(resampled, baseline)
        reordered = copy.copy(self.corrupt)
        reordered.candidate_ids = dict(reversed(list(self.corrupt.candidate_ids.items())))
        a = evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, "residual", scope="prompt_and_decode", donor=self.clean)
        b = evaluate(self.model, self.tokenizer, self.layout, reordered, 7, "residual", scope="prompt_and_decode", donor=self.clean)
        self.assertEqual(a, b)
        for expected, actual in zip(before, self.clean.past.to_legacy_cache()):
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_positive_greedy_alignment_and_eos_include_actual_recipient_prefix(self):
        import rha.specificity_scoring as scoring
        original = scoring.one_step_logits
        expected, calls = [ord("7"), ord("6"), ord("5"), 0], []

        def observed(model, token, past):
            calls.append((int(token.item()), cache_seq_len(past)))
            logits, cache = original(model, token, past)
            forced = logits.clone().fill_(-1000)
            forced[0, expected[(len(calls) - 1) // 2]] = 0
            return forced, cache

        def step(token, recipient, donor, index):
            return intervention_step(self.model, self.layout, token, recipient, donor, "residual", [], index, self.clean.length)

        with patch.object(scoring, "one_step_logits", side_effect=observed):
            start = step(self.corrupt.last_token, copy.deepcopy(self.corrupt.past), copy.deepcopy(self.clean.past), 0)
            result = continue_branch(self.model, self.tokenizer, start, step, 24)
        self.assertEqual(result["token_ids"], expected)
        self.assertEqual(result["stop"], "eos")
        self.assertEqual(calls[::2], calls[1::2])
        self.assertEqual([token for token, _ in calls[::2]], [int(self.corrupt.last_token.item())] + expected[:-1])
        self.assertEqual(len(result["steps"]), 4)

    def test_bad_positive_logits_alignment_and_hook_cleanup(self):
        import rha.specificity_scoring as scoring
        original, calls = scoring.one_step_logits, []

        def wrong_logits(*args):
            values, past = original(*args)
            calls.append(1)
            if len(calls) % 2 == 0:
                values = values.clone()
                values[0, 0] += 1
            return values, past

        with patch.object(scoring, "one_step_logits", side_effect=wrong_logits):
            with self.assertRaisesRegex(RuntimeError, "donor logits"):
                evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, "residual", scope="prompt_and_decode", donor=self.clean)
        self.assertFalse(self.model.model.norm._forward_pre_hooks)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            with final_residual(self.model):
                pass
        self.assertFalse(self.model.model.norm._forward_pre_hooks)
        bad = copy.copy(self.clean)
        bad.last_token = torch.tensor([[1]])
        with self.assertRaisesRegex(ValueError, "alignment"):
            evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, "residual", scope="prompt_and_decode", donor=bad)
        for action in ("zero", "resample"):
            with self.assertRaisesRegex(ValueError, "query-only"):
                evaluate(self.model, self.tokenizer, self.layout, self.clean, 7, action, self.pairs, "prompt_and_decode")


class SpecificityPipelineTests(unittest.TestCase, TinyFixture):
    @classmethod
    def setUpClass(cls):
        cls.initialize()
        cls.protocol = tiny_protocol()
        examples, references = build_examples(cls.tokenizer, cls.protocol, "smoke", set())
        cls.examples = [e for e in examples if e["variant"] == "literal"]
        cls.references = [r for r in references if r["variant"] == "literal"]
        cls.groups = {"address_L22H7": [(0, 0)], "random_address_L22H7_0": [(0, 2)],
                      "answer_address": [(0, 1), (1, 0)], "random_answer_address_0": [(0, 3), (1, 2)],
                      "non_address_core": [(1, 1)], "random_non_address_core_0": [(1, 3)]}
        refs = {r["reference_id"]: r for r in cls.references}
        cls.rows = [row for ex in cls.examples for row in iter_case_rows(cls.model, cls.tokenizer, cls.layout, ex,
                    refs.get(ex["reference_id"]), cls.groups, cls.protocol, 127)]
        cls.manifest = {"protocol": cls.protocol, "groups": cls.groups, "eos_token_ids": [0],
                        "n_independent_example_ids": 1, "n_reference_identities": 1, "n_reference_prompts": 5,
                        "n_example_pairs": 8, "rows_written": len(cls.rows), "expected_rows": 140}

    def test_complete_pipeline_positive_control_and_paired_interactions(self):
        result = validate_rows(self.manifest, self.examples, self.references, self.rows)
        self.assertEqual(result["rows"], 140)
        self.assertEqual(result["positive_control_cases"], 8)
        self.assertEqual(result["positive_max_logit_error"], 0)
        self.assertEqual(result["positive_max_token_score_error"], 0)
        self.assertTrue(summarize(self.rows)["by_condition"])
        contrasts = paired_analysis(self.manifest, self.rows, 30)["contrasts"]
        self.assertTrue(all(c["n_identities"] == 1 for c in contrasts))
        self.assertTrue(any("control_adjusted_interaction" in c["contrast"] for c in contrasts))

    def test_control_adjusted_difference_of_differences(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["delta_clean_logprob"] = 0
            if row["corruption"] != "different_first":
                continue
            if row["condition"] == "patch:prompt_only:address_L22H7":
                row["delta_clean_logprob"] = {"far_chat": 3, "near_chat": 1, "short_chat": .5}[row["view"]]
            elif row["condition"] == "patch:prompt_only:random_address_L22H7_0":
                row["delta_clean_logprob"] = {"far_chat": 1, "near_chat": .5, "short_chat": .25}[row["view"]]
            elif row["condition"] == "zero:prompt_only:non_address_core":
                row["delta_clean_logprob"] = {"far_chat": -1, "near_chat": -.25, "short_chat": 0, "far_raw": -5, "far_raw_space": -2}[row["view"]]
            elif row["condition"] == "zero:prompt_only:random_non_address_core_0":
                row["delta_clean_logprob"] = {"far_chat": 0, "near_chat": 0, "short_chat": 0, "far_raw": -1, "far_raw_space": -.5}[row["view"]]
        contrasts = {c["contrast"]: c for c in paired_analysis(self.manifest, rows, 30)["contrasts"] if c["metric"] == "delta_clean_logprob"}
        distance = contrasts["distance:far_chat_minus_near_chat:different_first:prompt_only:address_L22H7:control_adjusted_interaction"]
        boundary = contrasts["boundary:far_raw_space_minus_far_raw:zero:non_address_core:control_adjusted_interaction"]
        self.assertEqual((distance["mean"], distance["ci_low"], distance["ci_high"]), (1.5, 1.5, 1.5))
        self.assertEqual(boundary["mean"], 2.5)
        self.assertEqual(contrasts["task_ablation:far_chat_minus_near_chat:zero:non_address_core:control_adjusted_interaction"]["mean"], -.75)

    def test_grid_scores_donor_prefix_resample_source_and_positive_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "condition grid"):
            validate_rows(self.manifest, self.examples, self.references, self.rows[:-1])
        for field, value in (("donor_input_token_id", -1), ("verified_final_norm_calls", {}), ("donor_logit_max_abs_error", .1)):
            bad = copy.deepcopy(self.rows)
            row = next(r for r in bad if r["condition"] == "positive_transfer")
            row["candidate_scores"]["clean"]["steps"][1][field] = value
            with self.assertRaises(ValueError):
                validate_rows(self.manifest, self.examples, self.references, bad)
        bad = copy.deepcopy(self.rows)
        row = next(r for r in bad if r["condition"].startswith("resample:"))
        row["generation_steps"][0]["resample_reference_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "trace"):
            validate_rows(self.manifest, self.examples, self.references, bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["candidate_scores"]["clean"]["sum_logprob"] += 1
        with self.assertRaisesRegex(ValueError, "token sum"):
            validate_rows(self.manifest, self.examples, self.references, bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["generated_token_logprobs"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_rows(self.manifest, self.examples, self.references, bad)

    def test_prompt_and_reference_pairing_cannot_silently_change(self):
        bad = copy.deepcopy(self.examples)
        near = next(e for e in bad if e["view"] == "near_chat")
        near["prompt_tokens"] -= 1
        with self.assertRaises(ValueError):
            validate_design(self.manifest, bad, self.references)
        refs = copy.deepcopy(self.references)
        refs[0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_design(self.manifest, self.examples, refs)

    def test_smoke_gate_stale_environment_and_wrong_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = {"arguments": {"stage": "smoke", "chunk_size": 1024}, "protocol": {"revision": "abc"},
                        "source_sha256": {"file": "hash"}, "packages": {"torch": "version"},
                        "model_revision": "abc", "rows_sha256": "rows", "references_sha256": "refs"}
            (path / "manifest.json").write_text(json.dumps(manifest))
            with patch("scripts.analyze_publication_specificity.analyze", return_value={"validation": {"status": "passed"}}):
                args = (path, {"revision": "abc"}, {"file": "hash"}, {"torch": "version"}, 1024)
                self.assertEqual(check_smoke(*args)["references_sha256"], "refs")
                with self.assertRaisesRegex(ValueError, "stale"):
                    check_smoke(path, args[1], args[2], {"torch": "other"}, 1024)
                manifest["arguments"]["stage"] = "confirmation"
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "smoke-stage"):
                    check_smoke(*args)

    def test_partial_outputs_overwrites_and_missing_smoke_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                analyze(path)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                main(["--prepare-only", "--out-dir", str(path)])
        with self.assertRaisesRegex(ValueError, "requires --smoke-dir"):
            main(["--stage", "confirmation", "--out-dir", "logs/never_created_specificity_test"])

    def test_preparation_failure_is_recorded_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed_run"
            with patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("injected preparation failure")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    main(["--prepare-only", "--out-dir", str(path)])
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("injected preparation failure", manifest["error"])
            self.assertEqual(manifest["rows_written"], 0)
            with self.assertRaises(FileExistsError):
                main(["--prepare-only", "--out-dir", str(path)])


if __name__ == "__main__":
    unittest.main()
