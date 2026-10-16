import unittest
from pathlib import Path

from scripts.diagnose_publication_multifact import diagnose, summarize_condition

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts_publication/multifact_confirmation_8192"
SMOKE = ROOT / "artifacts_publication/multifact_smoke_8192"
EARLIER = ROOT / "artifacts_publication/specificity_confirmation_8192"


class MultiFactDiagnosticUnitTests(unittest.TestCase):
    def test_repeated_identities_are_not_extra_samples(self):
        with self.assertRaisesRegex(ValueError, "independent identity"):
            summarize_condition([{"example_id": "a"}, {"example_id": "a"}], {})

    def test_different_conditions_cannot_be_pooled(self):
        rows = [{"example_id": str(i), "view": "far_chat", "corruption": mode, "condition": "clean"}
                for i, mode in enumerate(("query_selection", "value_swap"))]
        with self.assertRaisesRegex(ValueError, "Do not pool"):
            summarize_condition(rows, {})


@unittest.skipUnless((RUN / "rows.jsonl.gz").exists() and (SMOKE / "rows.jsonl.gz").exists()
                     and (EARLIER / "rows.jsonl").exists(), "Requires committed results and extracted legacy specificity trace")
class MultiFactDiagnosticArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose(RUN, SMOKE, EARLIER)
        cls.conditions = {(r["view"], r["corruption"], r["condition"]): r for r in cls.result["conditions"] if r["family"] == "all"}
        cls.contrasts = {(c["contrast"], c["metric"]): c for c in cls.result["frozen_contrasts"]}

    def condition(self, view, mode, condition="patch:prompt_and_decode:answer_address"):
        return self.conditions[view, mode, condition]

    def test_integrity_and_positive_control_reproduce_errors_not_only_gold(self):
        audit = self.result["audit"]
        self.assertTrue(audit["all_saved_analyses_reproduce"])
        self.assertTrue(audit["smoke_gate_matches"])
        self.assertTrue(audit["both_compressed_and_uncompressed_hashes_verified"])
        self.assertEqual((audit["confirmation"]["rows"], audit["smoke"]["rows"]), (6880, 1376))
        self.assertEqual(audit["confirmation"]["independent_identities"], 40)
        self.assertEqual(audit["study_reference_stage_overlap"], 0)
        for stage, n, errors in (("confirmation", 160, 6), ("smoke", 32, 4)):
            positive = self.result["positive_control"][stage]
            self.assertEqual(positive["token_sequences_equal_actual_donor"], n)
            self.assertEqual(positive["stopping_conditions_equal_actual_donor"], n)
            self.assertEqual(positive["incorrect_donor_outputs_reproduced"], errors)
            self.assertEqual(positive["correct_target_outputs"], n - errors)
            self.assertEqual(positive["max_logit_error"], 0)
            self.assertEqual(positive["max_token_score_error"], 0)

    def test_transfer_is_not_preexisting_target_or_always_actual_donor(self):
        for view, mode, expected in (("far_chat", "value_swap", 29), ("far_chat", "query_selection", 32),
                                     ("near_chat", "value_swap", 35), ("near_chat", "query_selection", 38)):
            self.assertEqual(self.condition(view, mode)["count_exact_clean"], expected)
        swap, query = self.condition("far_chat", "value_swap"), self.condition("far_chat", "query_selection")
        self.assertEqual((swap["count_exact_clean"], swap["count_new_target_from_reference"]), (29, 28))
        self.assertEqual((query["count_exact_clean"], query["count_matches_actual_donor_sequence"]), (32, 34))
        self.assertEqual(query["count_new_target_from_reference"], 32)
        controls = self.result["patch_totals"]["confirmation"]
        self.assertEqual((controls["control_rows"], controls["control_exact_target_outputs"], controls["control_new_target_outputs"]), (3200, 20, 0))

    def test_primary_contrast_and_baseline_success_sensitivity(self):
        label = "patch:far_chat:query_selection:prompt_and_decode:answer_address_minus_controls"
        exact, score = self.contrasts[label, "delta_exact_clean"], self.contrasts[label, "delta_clean_logprob"]
        self.assertEqual(exact["n_identities"], 40)
        self.assertAlmostEqual(exact["mean"], .8)
        self.assertAlmostEqual(exact["ci_low"], .675)
        self.assertAlmostEqual(exact["ci_high"], .9)
        self.assertAlmostEqual(score["mean"], 2.22792, places=5)
        both = self.condition("far_chat", "query_selection")["both_baselines_correct"]
        self.assertEqual((both["n"], both["count_exact_target"]), (36, 31))
        common = {r["corruption"]: r for r in self.result["common_baseline_success_sensitivity"]}
        self.assertEqual(common["query_selection"]["count_exact_target"], {"far_chat": 31, "near_chat": 34})
        self.assertEqual(common["value_swap"]["count_exact_target"], {"far_chat": 28, "near_chat": 32})

    def test_task_accuracy_contrast_is_uncertain_and_near_effects_are_stronger(self):
        for view in ("far_chat", "near_chat"):
            label = f"task:query_minus_swap:{view}:prompt_and_decode:answer_address:control_adjusted"
            contrast = self.contrasts[label, "delta_exact_clean"]
            self.assertLess(contrast["ci_low"], 0)
            self.assertGreater(contrast["ci_high"], 0)
        for mode in ("value_swap", "query_selection"):
            label = f"distance:far_minus_near:{mode}:prompt_and_decode:answer_address:control_adjusted"
            self.assertLess(self.contrasts[label, "delta_clean_logprob"]["ci_high"], 0)
            self.assertLess(self.contrasts[label, "delta_exact_clean"]["ci_high"], 0)

    def test_scope_repairs_preserve_first_token_and_scores_remain_onset_dominated(self):
        transitions = {(r["view"], r["corruption"], r["group"]): r for r in self.result["scope_transitions"]}
        for view, mode, gained in (("far_chat", "query_selection", 0), ("far_chat", "value_swap", 2),
                                   ("near_chat", "query_selection", 4), ("near_chat", "value_swap", 7)):
            row = transitions[view, mode, "answer_address"]
            self.assertEqual(row["same_first_generated_token"], 40)
            self.assertEqual(row["new_exact_targets"], gained)
            self.assertEqual(row["lost_exact_targets"], 0)
            self.assertGreater(self.condition(view, mode)["first_digit_fraction_of_net_sequence_delta"], .9)
        for row in self.result["conditions"]:
            self.assertAlmostEqual(sum(row["mean_clean_token_logprob_delta"]), 6 * row["mean_delta_clean_logprob"])

    def test_position_heterogeneity_is_retained(self):
        positions = [r for r in self.result["target_position_breakdown"] if r["view"] == "far_chat" and r["corruption"] == "query_selection"]
        self.assertEqual([r["n"] for r in positions], [10] * 4)
        self.assertEqual([r["count_exact_target"] for r in positions], [10, 10, 3, 9])
        self.assertEqual([r["count_clean_baseline_correct"] for r in positions], [10, 10, 7, 10])

    def test_ablation_format_rank_content_and_censored_outputs_are_distinct(self):
        zero = self.condition("near_chat", "query_selection", "zero:prompt_only:non_address_core")
        resample = self.condition("near_chat", "query_selection", "resample:prompt_only:non_address_core")
        address = self.condition("near_chat", "query_selection", "resample:prompt_only:answer_address")
        self.assertEqual((zero["count_exact_clean"], zero["count_clean_number_content"]), (32, 40))
        self.assertEqual(resample["count_exact_clean"], 40)
        self.assertEqual((address["count_exact_clean"], address["count_clean_number_content"], address["count_clean_beats_all_candidates"]), (8, 10, 27))
        self.assertEqual(address["generation_stops"], {"eos": 38, "token_budget": 2})
        self.assertEqual(len(self.result["budget_outputs"]), 4)
        for row in self.result["conditions"]:
            if row["condition"].startswith("resample:") and row["condition"].split(":")[-1] in ("answer_address", "non_address_core"):
                self.assertEqual(row["count_contains_reference_number"], 0)

    def test_earlier_null_is_preserved_without_claiming_matched_cross_run_comparison(self):
        for comparison in self.result["earlier_comparison"]:
            self.assertEqual(comparison["comparison"], "descriptive_specificity_vs_multifact")
            self.assertEqual(comparison["old_count_exact_target"], 0)
            self.assertGreater(comparison["new_count_exact_target"], 0)
            self.assertEqual(comparison["old_recipient_contains_target"], 0)
            self.assertEqual(comparison["new_recipient_contains_target"], 40)


if __name__ == "__main__":
    unittest.main()
