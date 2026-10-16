import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from rha.multifact import (LABELS, body_parts, build_examples, condition_specs,
                           extra_metrics, identity_spec, reserved_ids, select_groups, token_metadata)
from rha.multifact_scoring import assert_control, capture_reference, evaluate, prepare_prompt
from rha.modeling import inspect_model_layout
from scripts.analyze_publication_multifact import (analyze, paired_analysis, read_compressed_rows, summarize,
                                                   validate_design, validate_rows, write_analysis)
from scripts.run_publication_confirmation import make_groups
from scripts.run_publication_multifact import check_smoke, iter_case_rows, main, source_hashes
from test_publication_specificity import TinyFixture
from test_publication_temporal import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]


def tiny_protocol():
    protocol = json.loads((ROOT / "configs/publication_multifact.json").read_text())
    protocol.update(target_tokens=1536, max_len=2048, max_new_tokens=3)
    return protocol


def design_manifest(protocol, examples, references):
    return {"protocol": protocol, "arguments": {"stage": "smoke"},
            "n_independent_example_ids": len({e["example_id"] for e in examples}),
            "n_reference_identities": len({r["seed"] for r in references}),
            "n_reference_prompts": len(references), "n_example_pairs": len(examples)}


class MultiFactPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.tokenizer = tiny_protocol(), CharacterTokenizer()
        cls.examples, cls.references = build_examples(cls.tokenizer, cls.protocol, "smoke", set())

    def test_frozen_controls_and_expected_run_sizes(self):
        original = json.loads((ROOT / "configs/publication_confirmation.json").read_text())
        groups = select_groups(self.protocol, original)
        old = make_groups(original, 5)
        self.assertEqual(len(groups), 18)
        self.assertTrue(all(pairs == old[name] for name, pairs in groups.items()))
        self.assertEqual((len(self.examples), len(self.references)), (32, 16))
        counts = Counter(len(condition_specs(e, groups, self.protocol)) for e in self.examples)
        self.assertEqual(counts, {31: 16, 55: 16})
        self.assertEqual(sum(size * count for size, count in counts.items()), 1376)
        full, refs = build_examples(self.tokenizer, self.protocol, "confirmation", set())
        self.assertEqual((len(full), len(refs)), (160, 80))
        self.assertEqual(sum(len(condition_specs(e, groups, self.protocol)) for e in full), 6880)

    def test_four_facts_query_only_swap_only_and_distance_pairing(self):
        self.assertEqual((self.examples, self.references), build_examples(self.tokenizer, self.protocol, "smoke", set()))
        manifest = design_manifest(self.protocol, self.examples, self.references)
        validate_design(manifest, self.examples, self.references)
        by_key = {(e["example_id"], e["view"], e["corruption"]): e for e in self.examples}
        for ex in self.examples:
            clean, document, question = body_parts(ex, ex["paragraphs"], ex["view"])
            corrupt, other_document, other_question = body_parts(ex, ex["paragraphs"], ex["view"], ex["corruption"])
            self.assertEqual(len(set(ex["candidate_answers"].values())), 4)
            for code in ex["codes"]:
                self.assertEqual(clean.count(code), 1)
                self.assertEqual(corrupt.count(code), 1)
            self.assertNotIn("[NEEDLE", clean)
            self.assertNotIn("Distractor note", clean)
            if ex["corruption"] == "query_selection":
                self.assertEqual(document, other_document)
                self.assertNotEqual(question, other_question)
            else:
                self.assertEqual(question, other_question)
                a, b = ex["clean_gold"], ex["corrupt_gold"]
                self.assertEqual(document.replace(a, "<swap>").replace(b, a).replace("<swap>", b), other_document)
            near = by_key[ex["example_id"], "near_chat", ex["corruption"]]
            far = by_key[ex["example_id"], "far_chat", ex["corruption"]]
            self.assertEqual(Counter(near["clean_prompt"]), Counter(far["clean_prompt"]))
            self.assertEqual(Counter(near["corrupt_prompt"]), Counter(far["corrupt_prompt"]))
            for code in ex["codes"]:
                self.assertGreater(far["source_spans"][code]["distance_to_question"], near["source_spans"][code]["distance_to_question"])

    def test_balanced_slots_and_disjoint_words_seeds_and_references(self):
        for stage, count in (("smoke", 2), ("confirmation", 10)):
            specs = [identity_spec(self.protocol, stage, family, seed) for family, seed in reserved_ids(self.protocol, stage)]
            self.assertEqual(Counter(s["target_slot"] for s in specs), {i: count for i in range(4)})
            self.assertEqual(Counter(s["alternate_slot"] for s in specs), {i: count for i in range(4)})
            self.assertTrue(all(s["target_slot"] != s["alternate_slot"] for s in specs))
        schedules = [reserved_ids(self.protocol, stage, ref) for stage in ("smoke", "confirmation") for ref in (False, True)]
        self.assertEqual(sum(map(len, schedules)), len({seed for schedule in schedules for _, seed in schedule}))
        self.assertNotEqual(self.protocol["stages"]["smoke"]["wording"], self.protocol["stages"]["confirmation"]["wording"])
        references = {r["reference_id"]: r for r in self.references}
        for ex in self.examples:
            if ex["reference_id"]:
                ref = references[ex["reference_id"]]
                self.assertEqual((ref["variant"], ref["view"]), (ex["variant"], ex["view"]))
                self.assertFalse(set(ref["codes"]) & set(ex["codes"]))

    def test_seed_overlap_truncation_bad_templates_and_duplicate_codes_fail_closed(self):
        for ref in (False, True):
            # Detect numerical seed reuse even if the old task family has a different name.
            seed = next(iter(reserved_ids(self.protocol, "smoke", ref)))[1]
            with self.assertRaisesRegex(ValueError, "overlaps"):
                build_examples(self.tokenizer, self.protocol, "smoke", {("old_family", seed)})
        bad = copy.deepcopy(self.protocol)
        bad["stages"]["confirmation"] = bad["stages"]["smoke"]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_examples(self.tokenizer, bad, "smoke", set())
        with self.assertRaisesRegex(ValueError, "truncation"):
            build_examples(self.tokenizer, dict(self.protocol, max_len=30), "smoke", set())
        with self.assertRaisesRegex(ValueError, "too short"):
            build_examples(self.tokenizer, dict(self.protocol, target_tokens=40), "smoke", set())
        ex = self.examples[0]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            token_metadata(self.tokenizer, ex["clean_prompt"] + ex["clean_gold"], ex["codes"], 2048)

    def test_four_candidate_margin_and_generation_metrics_are_distinct(self):
        answers = dict(zip(LABELS, ("123456", "234567", "345678", "456789")))
        scores = {key: {"sum_logprob": value} for key, value in zip(LABELS, (-3, -6, -7, -2))}
        result = extra_metrics(" 123456 and 456789", scores, answers, "987654")
        self.assertEqual(result["clean_best_other_margin"], -1)
        self.assertEqual(result["clean_beats_all_candidates"], 0)
        self.assertEqual(result["first_number_is_clean"], 1)
        self.assertEqual(result["verbatim_clean"], 0)
        self.assertEqual(result["contains_clean_number"], 1)
        self.assertEqual(extra_metrics("456789", scores, answers, None)["exact_other_fact"], 1)
        self.assertEqual(extra_metrics("987654", scores, answers, "987654")["exact_resample"], 1)
        self.assertEqual(extra_metrics("123456 987654", scores, answers, "987654")["resample_number_content"], 0)


class MultiFactScoringTests(unittest.TestCase, TinyFixture):
    @classmethod
    def setUpClass(cls):
        cls.initialize()
        cls.answers = {**cls.answers, "competitor_2": "345678"}
        cls.clean = prepare_prompt(cls.model, cls.tokenizer, cls.clean_text, cls.answers, 128, 7, 24)
        cls.corrupt = prepare_prompt(cls.model, cls.tokenizer, cls.corrupt_text, cls.answers, 128, 7, 24)

    def test_all_four_scores_match_independent_full_prefix_forwards(self):
        baseline = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7)
        self.assertEqual(set(baseline["candidate_scores"]), set(LABELS))
        for label, ids in self.clean.candidate_ids.items():
            for index, token in enumerate(ids):
                with torch.no_grad():
                    logits = self.model(input_ids=torch.tensor([self.tokenizer(self.clean_text).input_ids + ids[:index]]), use_cache=False).logits[0, -1]
                self.assertAlmostEqual(float(logits.float().log_softmax(-1)[token]), baseline["candidate_scores"][label]["token_logprobs"][index], places=5)

    def test_positive_transfer_matches_actual_donor_including_fourth_candidate(self):
        baseline = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7)
        positive = evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, "residual", scope="prompt_and_decode", donor=self.clean)
        assert_control(positive, baseline)
        self.assertNotEqual(baseline["generated_text"].strip(), self.answers["clean"])
        events = positive["generation_steps"] + [e for score in positive["candidate_scores"].values() for e in score["steps"]]
        self.assertTrue(all(e["donor_logit_max_abs_error"] == 0 for e in events))
        self.assertTrue(all(e["verified_final_norm_calls"] == {"donor": 1, "recipient": 1} for e in events))
        bad = copy.deepcopy(positive)
        bad["candidate_scores"]["competitor_2"]["token_logprobs"][4] += .1
        with self.assertRaisesRegex(RuntimeError, "candidate"):
            assert_control(bad, baseline)

    def test_positive_transfer_bfloat16_and_self_patches(self):
        model = copy.deepcopy(self.model).to(dtype=torch.bfloat16)
        layout = inspect_model_layout(model)
        clean = prepare_prompt(model, self.tokenizer, self.clean_text, self.answers, 128, 7, 7)
        corrupt = prepare_prompt(model, self.tokenizer, self.corrupt_text, self.answers, 128, 7, 7)
        baseline = evaluate(model, self.tokenizer, layout, clean, 7)
        positive = evaluate(model, self.tokenizer, layout, corrupt, 7, "residual", scope="prompt_and_decode", donor=clean)
        assert_control(positive, baseline)
        for scope in ("prompt_only", "prompt_and_decode"):
            self_patch = evaluate(model, self.tokenizer, layout, clean, 7, "patch", self.pairs, scope, clean)
            assert_control(self_patch, baseline)

    def test_candidate_order_cannot_change_greedy_branch_or_caches(self):
        before = [tuple(t.clone() for t in layer) for layer in self.corrupt.past.to_legacy_cache()]
        reordered = replace(self.corrupt, candidate_ids=dict(reversed(list(self.corrupt.candidate_ids.items()))))
        for action, scope in (("patch", "prompt_only"), ("patch", "prompt_and_decode"), ("residual", "prompt_and_decode")):
            pairs = self.pairs if action == "patch" else ()
            first = evaluate(self.model, self.tokenizer, self.layout, self.corrupt, 7, action, pairs, scope, self.clean)
            second = evaluate(self.model, self.tokenizer, self.layout, reordered, 7, action, pairs, scope, self.clean)
            self.assertEqual(first, second)
        for expected, actual in zip(before, self.corrupt.past.to_legacy_cache()):
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_resampling_self_is_identity_for_every_candidate(self):
        ref = {"reference_id": "self", "prompt": self.clean_text, "prompt_tokens": self.clean.length}
        captured = capture_reference(self.model, self.tokenizer, self.layout, ref, 128, 7)
        values = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7, "resample", self.pairs, "prompt_only", resample=captured)
        baseline = evaluate(self.model, self.tokenizer, self.layout, self.clean, 7)
        assert_control(values, baseline)
        self.assertTrue(all(not step["patch_applied"] for step in values["candidate_scores"]["competitor_2"]["steps"][1:]))

    def test_missing_duplicate_or_bad_length_candidates_fail(self):
        for answers in ({k: v for k, v in self.answers.items() if k != "competitor_2"},
                        dict(self.answers, competitor_2=self.answers["clean"]), dict(self.answers, competitor_2="12345")):
            with self.assertRaises(ValueError):
                prepare_prompt(self.model, self.tokenizer, self.clean_text, answers, 128, 7, 7)


class MultiFactPipelineTests(unittest.TestCase, TinyFixture):
    @classmethod
    def setUpClass(cls):
        cls.initialize()
        cls.protocol = tiny_protocol()
        all_examples, all_refs = build_examples(cls.tokenizer, cls.protocol, "smoke", set())
        cls.examples = all_examples[:4]
        cls.references = all_refs[:2]
        cls.groups = {"address_L22H7": [(0, 0)], "random_address_L22H7_0": [(0, 1)],
                      "answer_address": [(0, 0), (1, 0)], "random_answer_address_0": [(0, 2), (1, 2)],
                      "non_address_core": [(1, 1)], "random_non_address_core_0": [(1, 3)]}
        refs = {r["reference_id"]: r for r in cls.references}
        cls.rows = [row for ex in cls.examples for row in iter_case_rows(cls.model, cls.tokenizer, cls.layout, ex,
                    refs.get(ex["reference_id"]), cls.groups, cls.protocol, 127)]
        cls.manifest = {**design_manifest(cls.protocol, cls.examples, cls.references), "groups": cls.groups,
                        "eos_token_ids": [0], "rows_written": len(cls.rows), "expected_rows": 76}

    def test_complete_pipeline_and_independent_identity_count(self):
        result = validate_rows(self.manifest, self.examples, self.references, self.rows)
        self.assertEqual((result["rows"], result["positive_control_cases"]), (76, 4))
        self.assertEqual(result["positive_max_logit_error"], 0)
        self.assertEqual(result["positive_max_token_score_error"], 0)
        self.assertTrue(summarize(self.rows)["by_condition"])
        contrasts = paired_analysis(self.manifest, self.rows, 30)["contrasts"]
        self.assertTrue(all(c["n_identities"] == 1 for c in contrasts))
        self.assertTrue(any(c["contrast"].startswith("task_distance:") for c in contrasts))

    def test_control_adjusted_paired_task_distance_contrast(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["delta_clean_logprob"] = 0
            if row["condition"] == "patch:prompt_only:address_L22H7":
                row["delta_clean_logprob"] = {("far_chat", "query_selection"): 8, ("near_chat", "query_selection"): 3,
                                               ("far_chat", "value_swap"): 3, ("near_chat", "value_swap"): 1}[row["view"], row["corruption"]]
            elif row["condition"] == "patch:prompt_only:random_address_L22H7_0":
                row["delta_clean_logprob"] = 2 if row["view"] == "far_chat" else 1
        contrasts = {c["contrast"]: c for c in paired_analysis(self.manifest, rows, 30)["contrasts"] if c["metric"] == "delta_clean_logprob"}
        self.assertEqual(contrasts["distance:far_minus_near:query_selection:prompt_only:address_L22H7:control_adjusted"]["mean"], 4)
        self.assertEqual(contrasts["task:query_minus_swap:far_chat:prompt_only:address_L22H7:control_adjusted"]["mean"], 5)
        interaction = contrasts["task_distance:query_minus_swap:far_minus_near:prompt_only:address_L22H7:control_adjusted"]
        self.assertEqual((interaction["mean"], interaction["ci_low"], interaction["ci_high"]), (3, 3, 3))

    def test_missing_grid_scores_and_fourth_branch_traces_fail_closed(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            with self.assertRaisesRegex(ValueError, "condition grid"):
                validate_rows(self.manifest, self.examples, self.references, rows)
        for field, value in (("donor_input_token_id", -1), ("verified_final_norm_calls", {}), ("donor_logit_max_abs_error", .1)):
            bad = copy.deepcopy(self.rows)
            row = next(r for r in bad if r["condition"] == "positive_transfer")
            row["candidate_scores"]["competitor_2"]["steps"][1][field] = value
            with self.assertRaises(ValueError):
                validate_rows(self.manifest, self.examples, self.references, bad)
        bad = copy.deepcopy(self.rows)
        next(r for r in bad if r["condition"].startswith("resample:"))["generation_steps"][0]["resample_reference_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "trace"):
            validate_rows(self.manifest, self.examples, self.references, bad)
        for field, value in (("sum_logprob", 100), ("token_logprobs", [float("nan")] * 6)):
            bad = copy.deepcopy(self.rows)
            bad[0]["candidate_scores"]["competitor_2"][field] = value
            with self.assertRaises(ValueError):
                validate_rows(self.manifest, self.examples, self.references, bad)

    def test_damaged_design_mapping_hash_reference_and_distance_fail(self):
        for change in (lambda ex: ex[0]["candidate_answers"].update(competitor_2="999999"),
                       lambda ex: ex[0].update(clean_prompt_sha256="bad"),
                       lambda ex: ex[1].update(reference_id="wrong"),
                       lambda ex: ex[2].update(prompt_tokens=100),
                       lambda ex: ex[0]["changed_token_positions"].append(0)):
            bad = copy.deepcopy(self.examples)
            change(bad)
            with self.assertRaises(ValueError):
                validate_design(self.manifest, bad, self.references)

    def test_compressed_rows_roundtrip_and_both_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            raw = ("\n".join(json.dumps(row) for row in self.rows) + "\n").encode()
            archive = gzip.compress(raw)
            (path / "rows.jsonl.gz").write_bytes(archive)
            hashes = {"rows_sha256": hashlib.sha256(raw).hexdigest(), "rows_gzip_sha256": hashlib.sha256(archive).hexdigest()}
            self.assertEqual(read_compressed_rows(path, hashes), self.rows)
            for key in hashes:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    read_compressed_rows(path, dict(hashes, **{key: "bad"}))

    def test_complete_artifact_audit_and_saved_analysis_reproduce(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = {**self.manifest, "status": "complete", "git_commit": "a" * 40,
                        "original_protocol": {"test_fixture": True}, "protocol_source": "protocol.json",
                        "original_protocol_source": "original.json", "source_sha256": {"protocol.json": "p", "original.json": "o"},
                        "model_revision": self.protocol["revision"],
                        "tokenizer_and_chat_template_sha256": self.protocol["tokenizer_and_chat_template_sha256"]}
            for name, records in (("examples", self.examples), ("references", self.references), ("rows", self.rows)):
                raw = ("\n".join(json.dumps(row) for row in records) + "\n").encode()
                manifest[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
                if name == "rows":
                    archive = gzip.compress(raw)
                    (path / "rows.jsonl.gz").write_bytes(archive)
                    manifest["rows_gzip_sha256"] = hashlib.sha256(archive).hexdigest()
                else:
                    (path / (name + ".jsonl")).write_bytes(raw)
            (path / "manifest.json").write_text(json.dumps(manifest))

            def committed(command, **kwargs):
                return json.dumps(manifest["protocol"] if command[-1].endswith(":protocol.json") else manifest["original_protocol"]).encode()

            def schedule(protocol, stage, references):
                records = self.references if references else self.examples
                return {(r["variant"], r["seed"]) for r in records}

            module = "scripts.analyze_publication_multifact."
            with patch(module + "verify_committed_sources"), patch(module + "subprocess.check_output", side_effect=committed), \
                    patch(module + "select_groups", return_value=self.groups), patch(module + "reserved_ids", side_effect=schedule), \
                    patch(module + "prior_ids", return_value=set()):
                result = analyze(path, 30)
                write_analysis(path, result)
                self.assertEqual(analyze(path, 30), result)
                saved = json.loads((path / "paired_analysis.json").read_text())
                saved["contrasts"][0]["mean"] += 1
                (path / "paired_analysis.json").write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, "paired_analysis does not reproduce"):
                    analyze(path, 30)
                write_analysis(path, result)
                with patch(module + "subprocess.check_output", return_value=b"{}"):
                    with self.assertRaisesRegex(ValueError, "committed protocol"):
                        analyze(path, 30)

    def test_source_inventory_includes_adapter_and_frozen_engines(self):
        hashes = source_hashes(ROOT / "configs/publication_multifact.json", ROOT / "configs/publication_confirmation.json")
        for name in ("rha/multifact.py", "rha/multifact_scoring.py", "rha/temporal_scoring.py", "rha/specificity_scoring.py",
                     "scripts/run_publication_multifact.py", "scripts/analyze_publication_multifact.py"):
            self.assertEqual(hashes[name], hashlib.sha256((ROOT / name).read_bytes()).hexdigest())

    def test_smoke_gate_rejects_stale_sources_packages_chunk_and_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = {"arguments": {"stage": "smoke", "chunk_size": 1024}, "protocol": {"revision": "abc"},
                        "source_sha256": {"file": "hash"}, "packages": {"torch": "version"}, "model_revision": "abc",
                        "rows_sha256": "rows", "references_sha256": "refs"}
            (path / "manifest.json").write_text(json.dumps(manifest))
            with patch("scripts.analyze_publication_multifact.analyze", return_value={"validation": {"status": "passed"}}):
                args = (path, manifest["protocol"], manifest["source_sha256"], manifest["packages"], 1024)
                self.assertEqual(check_smoke(*args)["rows_sha256"], "rows")
                for index, replacement in ((1, {"revision": "other"}), (2, {"file": "bad"}), (3, {"torch": "other"}), (4, 512)):
                    bad = list(args)
                    bad[index] = replacement
                    with self.assertRaises(ValueError):
                        check_smoke(*bad)
                manifest["arguments"]["stage"] = "confirmation"
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "smoke-stage"):
                    check_smoke(*args)

    def test_partial_outputs_overwrites_and_missing_smoke_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                analyze(path)
            with self.assertRaises(FileExistsError):
                main(["--prepare-only", "--out-dir", str(path)])
        with self.assertRaisesRegex(ValueError, "requires --smoke-dir"):
            main(["--stage", "confirmation", "--out-dir", "logs/never_created_multifact_test"])

    def test_failed_preparation_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed"
            with patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    main(["--prepare-only", "--out-dir", str(path)])
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual((manifest["status"], manifest["rows_written"]), ("failed", 0))
            self.assertIn("injected", manifest["error"])


if __name__ == "__main__":
    unittest.main()
