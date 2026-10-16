import unittest
from pathlib import Path

from scripts.diagnose_publication_specificity import diagnose, number_diagnostics, summarize_condition

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts_publication/specificity_confirmation_8192"
SMOKE = ROOT / "artifacts_publication/specificity_smoke_8192"
TEMPORAL = ROOT / "artifacts_publication/temporal_confirmation_8192_chat"
EARLIER = ROOT / "artifacts_publication/confirmation_8192_chat"


class SpecificityDiagnosticUnitTests(unittest.TestCase):
    def test_first_number_is_not_exclusive_content_or_exact_answer(self):
        row = {"clean_gold": "123456", "unrelated_decoy": "987654", "resample_gold": "654321",
               "generated_text": "123456. The irrelevant number is 987654."}
        result = number_diagnostics(row)
        self.assertEqual(result["first_number_is_clean"], 1)
        self.assertEqual(result["contains_clean_number"], 1)
        self.assertEqual(result["mentions_decoy"], 1)
        row["generated_text"] = "The answer is 654321, not 123456."
        self.assertEqual(number_diagnostics(row)["first_number_is_reference"], 1)
        self.assertEqual(number_diagnostics(row)["contains_reference_number"], 1)
        self.assertEqual(number_diagnostics(row)["first_number_is_clean"], 0)
        row["generated_text"] = "1234567"
        self.assertEqual(number_diagnostics(row)["contains_clean_number"], 0)

    def test_repeated_views_are_not_independent_identities(self):
        with self.assertRaisesRegex(ValueError, "independent identity"):
            summarize_condition([{"example_id": "a"}, {"example_id": "a"}], {})


@unittest.skipUnless(all((p / "rows.jsonl").exists() for p in (RUN, SMOKE, TEMPORAL, EARLIER)),
                     "Requires committed artifacts and losslessly extracted full specificity trace")
class SpecificityDiagnosticArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose(RUN, SMOKE, TEMPORAL, EARLIER)
        cls.conditions = {(r["view"], r["corruption"], r["condition"], r["variant"]): r for r in cls.result["conditions"]}

    def condition(self, view, name):
        return self.conditions[view, "different_first", name, "all"]

    def test_integrity_positive_control_and_censored_outputs_are_retained(self):
        audit = self.result["audit"]
        self.assertTrue(audit["all_saved_analyses_reproduce"])
        self.assertTrue(audit["compressed_trace_verified"])
        self.assertEqual(audit["confirmation"]["independent_identities"], 40)
        self.assertEqual(audit["confirmation"]["rows"], 12640)
        self.assertEqual(audit["smoke"]["rows"], 1580)
        for stage, n in (("confirmation", 320), ("smoke", 40)):
            positive = self.result["positive_control"][stage]
            self.assertEqual(positive["generated_token_sequences_equal_donor"], n)
            self.assertEqual(positive["stopping_conditions_equal_donor"], n)
            self.assertEqual(positive["max_logit_error"], 0)
            self.assertEqual(positive["max_token_score_error"], 0)
        self.assertEqual(self.result["positive_control"]["confirmation"]["donor_budget_stops_reproduced"], 20)

    def test_no_full_sparse_transfer_despite_positive_calibration(self):
        patches = self.result["all_patches"]["confirmation"]
        self.assertEqual(patches["condition_rows_not_independent_trials"], 5760)
        self.assertEqual(patches["count_exact_clean"], 0)
        self.assertEqual(patches["count_clean_number_content"], 0)
        self.assertEqual(patches["count_contains_clean_number"], 0)
        self.assertLess(patches["maximum_answer_logprob_margin"], 0)
        for view in ("far_chat", "near_chat", "short_chat"):
            self.assertEqual(self.condition(view, "positive_transfer")["count_exact_clean"], 40)

    def test_core_boundary_penalty_is_not_initial_number_loss(self):
        raw = self.condition("far_raw", "zero:prompt_only:non_address_core")
        space = self.condition("far_raw_space", "zero:prompt_only:non_address_core")
        self.assertEqual(raw["count_first_number_is_clean"], 40)
        self.assertEqual(raw["count_clean_number_content"], 33)
        self.assertEqual(raw["generation_stops"], {"token_budget": 40})
        self.assertLess(raw["mean_delta_clean_logprob"], -1)
        self.assertGreater(space["mean_delta_clean_logprob"], -.01)
        self.assertEqual(space["count_first_number_is_clean"], 40)
        for row in self.result["conditions"]:
            self.assertAlmostEqual(sum(row["mean_clean_token_logprob_delta"]), 6 * row["mean_delta_clean_logprob"])

    def test_zero_and_resample_have_different_behavioral_effects(self):
        for view, exact_zero, exact_resample in (("far_chat", 30, 15), ("near_chat", 30, 20), ("short_chat", 40, 13)):
            core_zero = self.condition(view, "zero:prompt_only:non_address_core")
            core_resample = self.condition(view, "resample:prompt_only:non_address_core")
            address_zero = self.condition(view, "zero:prompt_only:answer_address")
            address_resample = self.condition(view, "resample:prompt_only:answer_address")
            self.assertEqual(core_zero["count_exact_clean"], exact_zero)
            self.assertEqual(core_zero["count_clean_number_content"], 40)
            self.assertEqual(core_resample["count_exact_clean"], 40)
            self.assertEqual(address_zero["count_exact_clean"], 40)
            self.assertEqual(address_resample["count_exact_clean"], exact_resample)
            self.assertEqual(address_resample["count_exact_resample"], 0)
            self.assertEqual(address_resample["count_contains_reference_number"], 0)

    def test_distance_effects_are_not_uniformly_far_dominant(self):
        contrasts = {(c["contrast"], c["metric"]): c for c in self.result["primary_contrasts"]}
        base = "distance:far_chat_minus_near_chat:different_first:"
        prompt = contrasts[base + "prompt_only:answer_address:control_adjusted_interaction", "delta_clean_logprob"]
        combined = contrasts[base + "prompt_and_decode:answer_address:control_adjusted_interaction", "delta_clean_logprob"]
        self.assertGreater(prompt["ci_low"], 0)
        self.assertLess(combined["ci_low"], 0)
        self.assertGreater(combined["ci_high"], 0)
        short = contrasts["distance:far_chat_minus_short_chat:different_first:prompt_and_decode:answer_address:control_adjusted_interaction", "delta_clean_logprob"]
        self.assertLess(short["ci_high"], 0)


if __name__ == "__main__":
    unittest.main()
