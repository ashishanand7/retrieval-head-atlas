import copy
import json
import unittest
from pathlib import Path

from scripts.diagnose_publication_confirmation import diagnose, validate_design

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts_publication/confirmation_8192_chat"
PILOT = ROOT / "artifacts_publication/pilot_4096_chat"


@unittest.skipUnless(RUN.exists() and PILOT.exists(), "Requires the committed confirmation and pilot results")
class ConfirmationDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = diagnose(RUN, PILOT)
        cls.conditions = {(row["marker"], row["condition"], row["variant"]): row for row in cls.result["conditions"]}

    def test_integrity_and_no_complete_answer_transfer(self):
        self.assertEqual(self.result["audit"]["rows"], 16640)
        self.assertEqual(self.result["audit"]["independent_identities"], 160)
        self.assertEqual(self.result["audit"]["pilot_identity_overlap"], 0)
        self.assertEqual(self.result["audit"]["self_patch_max_score_error"], 0.0)
        patches = self.result["all_patches"]
        self.assertEqual(patches["condition_rows_not_independent_trials"], 7680)
        self.assertEqual(patches["exact_donor_outputs"], 0)
        self.assertEqual(patches["donor_content_diagnostic_outputs"], 0)
        self.assertLess(patches["maximum_full_donor_minus_recipient_margin"], 0)

    def test_baselines_and_single_head_ablations_preserve_exact_answers(self):
        for marker in ("marked", "unmarked"):
            for condition in ("clean", "ablate:address_L22H7", "ablate:support_L20H7"):
                self.assertEqual(self.conditions[marker, condition, "all"]["count_exact_clean"], 160)
            self.assertEqual(self.conditions[marker, "corrupt", "all"]["count_exact_corrupt"], 160)

    def test_content_loss_is_not_format_loss(self):
        for marker, strict, content in (("marked", 141, 158), ("unmarked", 137, 156)):
            core = self.conditions[marker, "ablate:non_address_core", "all"]
            self.assertEqual(core["count_exact_clean"], strict)
            self.assertEqual(core["count_clean_number_content"], content)
            self.assertEqual(core["count_content_preserved_nonexact"], content - strict)
            self.assertEqual(core["count_no_clean_number_content"], 160 - content)
        alias = self.conditions["unmarked", "ablate:answer_address", "alias"]
        self.assertEqual(alias["count_exact_clean"], 25)
        self.assertEqual(alias["count_exact_unrelated_decoy"], 7)
        literal = self.conditions["unmarked", "ablate:non_address_core", "literal"]
        self.assertEqual(literal["count_clean_number_content"], 32)
        self.assertEqual(literal["count_exact_clean"], 15)

    def test_onset_effect_and_shared_first_token_subgroup(self):
        row = self.conditions["unmarked", "patch:address_L22H7", "all"]
        self.assertAlmostEqual(row["mean_delta_clean_logprob"], 1.0382876923)
        self.assertGreater(row["mean_first_token_delta"], row["mean_sequence_delta"])
        self.assertLess(row["mean_remaining_tokens_delta"], 0)
        self.assertAlmostEqual(row["mean_sequence_delta"], row["mean_first_token_delta"] + row["mean_remaining_tokens_delta"])
        subgroups = [row for row in self.result["first_token_subgroups"]
                     if row["marker"] == "unmarked" and row["condition"] == "patch:answer_address"]
        same = next(row for row in subgroups if not row["first_answer_tokens_differ"])
        different = next(row for row in subgroups if row["first_answer_tokens_differ"])
        self.assertEqual(same["n_identities"], 19)
        self.assertEqual(different["n_identities"], 141)
        self.assertEqual(different["count_first_token_matches_clean_baseline"], 111)
        self.assertEqual(different["count_exact_clean"], 0)

    def test_active_and_repeated_controls_are_retained(self):
        controls = {(row["marker"], row["focal_group"]): row for row in self.result["controls"]}
        support = controls["unmarked", "support_L20H7"]
        self.assertEqual(support["draw_count"], 5)
        self.assertEqual(support["distinct_head_sets"], 4)
        core = controls["unmarked", "non_address_core"]
        self.assertGreater(core["mean_patch_delta"], self.conditions["unmarked", "patch:non_address_core", "all"]["mean_delta_clean_logprob"])
        large = next(row for row in core["draws"] if row["group"].endswith("_3"))
        self.assertIn([22, 7], large["address_overlap"])

    def test_changed_design_is_rejected(self):
        manifest = json.loads((RUN / "manifest.json").read_text())
        examples = [json.loads(line) for line in (RUN / "examples.jsonl").read_text().splitlines()]
        bad = copy.deepcopy(manifest)
        bad["groups"]["random_support_L20H7_4"] = [[20, 0]]
        with self.assertRaisesRegex(ValueError, "frozen control sampler"):
            validate_design(bad, examples)
        examples[0]["clean_prompt"] += "extra text"
        with self.assertRaisesRegex(ValueError, "outside the answer"):
            validate_design(manifest, examples)


if __name__ == "__main__":
    unittest.main()
