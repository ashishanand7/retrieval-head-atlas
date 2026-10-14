import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.compare_publication_pilots import compare, content_diagnostics, load_checked

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts_publication/pilot_4096_raw"
CHAT = ROOT / "artifacts_publication/pilot_4096_chat"


class ContentDiagnosticTests(unittest.TestCase):
    def test_repetition_is_distinguished_from_wrong_content(self):
        self.assertEqual(content_diagnostics("123456. The code is 123456.", "123456"),
                         {"first_number_matches": True, "only_distinct_number_matches": True})
        self.assertEqual(content_diagnostics("123456 or 654321", "123456"),
                         {"first_number_matches": True, "only_distinct_number_matches": False})
        self.assertEqual(content_diagnostics("12345", "123456"),
                         {"first_number_matches": False, "only_distinct_number_matches": False})


@unittest.skipUnless(RAW.exists() and CHAT.exists(), "Requires the committed pilot results")
class PilotComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = compare(RAW, CHAT)
        cls.conditions = {(row["format"], row["marker"], row["condition"]): row for row in cls.result["conditions"]}

    def test_integrity_and_pairing(self):
        audit = self.result["audit"]
        self.assertEqual(audit["rows_per_format"], 400)
        self.assertEqual(audit["independent_identities_shared_across_formats_and_markers"], 10)
        self.assertEqual(audit["self_patch_max_score_error"], {"raw": 0.0, "chat": 0.0})
        self.assertEqual(audit["historical_identity_overlap"], 0)

    def test_format_and_content_are_not_conflated(self):
        raw = self.conditions["raw", "marked", "clean"]
        self.assertEqual(raw["count_exact_clean"], 0)
        self.assertEqual(raw["count_clean_only_distinct_number_matches"], 10)
        for marker in ("marked", "unmarked"):
            self.assertEqual(self.conditions["chat", marker, "clean"]["count_exact_clean"], 10)
            self.assertEqual(self.conditions["chat", marker, "patch:address_L22H7"]["count_exact_clean"], 0)
        core = self.conditions["chat", "unmarked", "ablate:non_address_core"]
        self.assertEqual(core["count_exact_clean"], 7)
        self.assertEqual(core["count_clean_only_distinct_number_matches"], 9)

    def test_effects_and_first_token_decomposition(self):
        effect = self.conditions["chat", "unmarked", "patch:address_L22H7"]["mean_delta_clean_logprob"]
        self.assertAlmostEqual(effect, 0.862395, places=6)
        raw_core = next(row for row in self.result["token_decomposition"]
                        if (row["format"], row["marker"], row["condition"]) == ("raw", "marked", "ablate:non_address_core"))
        self.assertGreater(raw_core["first_over_total_ratio"], 0.998)
        self.assertAlmostEqual(raw_core["mean_sequence_logprob_delta"],
                               raw_core["mean_first_token_delta"] + raw_core["mean_remaining_tokens_sum_delta"])

    def test_corrupted_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "raw"
            shutil.copytree(RAW, copy)
            summary = json.loads((copy / "summary.json").read_text())
            summary["n_rows"] = 399
            (copy / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "summary"):
                load_checked(copy)


if __name__ == "__main__":
    unittest.main()
