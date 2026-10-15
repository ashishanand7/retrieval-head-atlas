import unittest
from pathlib import Path

from scripts.diagnose_publication_temporal import diagnose, donor_prefix, summarize_condition

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts_publication/temporal_confirmation_8192_chat"
SMOKE = ROOT / "artifacts_publication/temporal_smoke_8192_chat"
EARLIER = ROOT / "artifacts_publication/confirmation_8192_chat"


class TemporalDiagnosticUnitTests(unittest.TestCase):
    def test_prefix_match_does_not_imply_exact_answer(self):
        row = {"candidate_token_ids": {"clean": [1, 2, 3, 4, 5, 6]}, "generated_token_ids": [1, 2, 3, 4, 5, 6, 9]}
        self.assertEqual(donor_prefix(row, 6), 1)
        row["generated_token_ids"] = [1, 2]
        self.assertEqual(donor_prefix(row, 2), 1)
        self.assertEqual(donor_prefix(row, 3), 0)
        for length in (0, 7):
            with self.assertRaises(ValueError):
                donor_prefix(row, length)

    def test_cannot_pool_corruptions_as_independent_identities(self):
        with self.assertRaisesRegex(ValueError, "independent identity"):
            summarize_condition([{"example_id": "a"}, {"example_id": "a"}], {})


@unittest.skipUnless(all(p.exists() for p in (RUN, SMOKE, EARLIER)), "Requires committed publication artifacts")
class TemporalDiagnosticArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose(RUN, SMOKE, EARLIER)
        cls.conditions = {(r["corruption"], r["condition"], r["variant"]): r for r in cls.result["conditions"]}

    def test_integrity_and_baseline_error_are_retained(self):
        audit = self.result["audit"]
        self.assertEqual(audit["confirmation"]["rows"], 3520)
        self.assertEqual(audit["confirmation"]["independent_identities"], 40)
        self.assertEqual(audit["smoke"]["independent_identities"], 5)
        self.assertEqual(audit["prior_identity_overlap"], 0)
        self.assertEqual(audit["confirmation"]["self_patch_max_token_score_error"], 0)
        failures = self.result["baseline_failures"]
        self.assertEqual(len(failures), 2)
        self.assertEqual({r["example_id"] for r in failures}, {"literal:11100006"})
        self.assertEqual({r["generated_text"] for r in failures}, {"820345"})
        self.assertTrue(all(r["n_identities"] == 39 for r in self.result["baseline_success_sensitivity"]))

    def test_no_complete_transfer_in_any_patch(self):
        patches = self.result["all_patches"]
        self.assertEqual(patches["condition_rows_not_independent_trials"], 2880)
        for metric in ("exact_clean", "sole_number_is_clean", "clean_number_content"):
            self.assertEqual(patches[f"count_{metric}"], 0)
        self.assertLess(patches["maximum_answer_logprob_margin"], 0)
        self.assertEqual(patches["control_outputs_changed_from_corrupt"], 0)

    def test_later_scores_can_change_without_changing_greedy_output(self):
        for group in ("address_L22H7", "answer_address"):
            row = self.conditions["same_first", f"patch:decode_only:{group}", "all"]
            self.assertEqual(row["count_exact_corrupt"], 40)
            self.assertEqual(row["count_first_differing_token_is_clean"], 0)
            self.assertGreater(row["mean_clean_token_logprob_delta"][1], 3)
            self.assertLess(row["maximum_first_differing_token_probability_clean"], .003)
            self.assertEqual(row["mean_clean_token_logprob_delta"][0], 0)
        for row in self.result["conditions"]:
            self.assertAlmostEqual(sum(row["mean_clean_token_logprob_delta"]), 6 * row["mean_delta_clean_logprob"])

    def test_prefix_gains_and_scope_transitions_are_not_complete_transfer(self):
        prompt = self.conditions["different_first", "patch:prompt_only:answer_address", "all"]
        both = self.conditions["different_first", "patch:prompt_and_decode:answer_address", "all"]
        self.assertEqual(prompt["count_donor_prefix_at_least"], [31, 5, 3, 0, 0, 0])
        self.assertEqual(both["count_donor_prefix_at_least"], [31, 19, 6, 0, 0, 0])
        self.assertEqual(both["count_donor_prefix_at_least_and_exact_decoy"], [4, 2, 2, 0, 0, 0])
        self.assertEqual(both["count_exact_decoy"], 4)
        transitions = self.result["scope_transitions"]
        self.assertTrue(all(r["n_changed"] == 0 for r in transitions if r["right"] == "corrupt"))
        both_vs_prompt = {(r["corruption"], r["group"]): r["n_changed"] for r in transitions if r["right"] != "corrupt"}
        self.assertEqual(both_vs_prompt["different_first", "answer_address"], 20)
        self.assertEqual(both_vs_prompt["different_first", "address_L22H7"], 4)
        self.assertEqual(both_vs_prompt["same_first", "answer_address"], 0)


if __name__ == "__main__":
    unittest.main()
