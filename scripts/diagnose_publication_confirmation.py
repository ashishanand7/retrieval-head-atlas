#!/usr/bin/env python
"""Audit the held-out confirmation and separate likelihood, onset, and content effects."""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.confirmation import VARIANTS
from rha.prompts import NEEDLE_END, NEEDLE_START, seeded_decoy
from scripts.analyze_publication_confirmation import paired_interval
from scripts.compare_publication_pilots import content_diagnostics, load_checked, sha256
from scripts.run_publication_confirmation import make_groups


def row_diagnostics(row, lookup):
    baseline = lookup[row["example_id"], row["marker_mode"], row["reference"]]
    clean = lookup[row["example_id"], row["marker_mode"], "clean"]
    content = content_diagnostics(row["generated_text"], row["clean_gold"])["only_distinct_number_matches"]
    total = row["answer_tokens"] * row["delta_clean_logprob"]
    first = row["first_clean_logprob"] - baseline["first_clean_logprob"]
    return {
        "sequence_delta": total, "first_token_delta": first, "remaining_tokens_delta": total - first,
        "clean_number_content": int(content),
        "content_preserved_nonexact": int(content and not row["exact_clean"]),
        "no_clean_number_content": int(not content),
        "exact_unrelated_decoy": int(row["generated_text"].strip() == seeded_decoy(row["seed"])),
        "changed_output": int(row["generated_text"] != baseline["generated_text"]),
        "first_token_matches_clean_baseline": int(row["generated_token_ids"][0] == clean["generated_token_ids"][0]),
        "positive_full_answer_margin": int(row["answer_logprob_margin"] > 0),
        "positive_first_answer_margin": int(row["first_answer_logprob_margin"] > 0),
        "positive_likelihood_change": int(row["delta_clean_logprob"] > 0),
        "token_budget_stop": int(row["generation_stop"] == "token_budget"),
    }


def summarize_condition(rows, lookup, intervals=False):
    diagnostics = [row_diagnostics(row, lookup) for row in rows]
    result = {"n_identities": len(rows)}
    for key in ("clean_mean_logprob", "delta_clean_logprob", "delta_answer_margin", "answer_logprob_margin"):
        result[f"mean_{key}"] = statistics.fmean(row[key] for row in rows)
    for key in ("exact_clean", "exact_corrupt", "sole_number_is_clean", "sole_number_is_corrupt", "six_digits_only"):
        result[f"count_{key}"] = sum(row[key] for row in rows)
    for key in diagnostics[0]:
        if key.endswith("_delta"):
            result[f"mean_{key}"] = statistics.fmean(row[key] for row in diagnostics)
        else:
            result[f"count_{key}"] = sum(row[key] for row in diagnostics)
    if intervals:
        result["delta_clean_logprob_interval"] = paired_interval(
            [row["delta_clean_logprob"] for row in rows], [row["variant"] for row in rows],
        )
    return result


def validate_design(manifest, examples):
    # Rebuild the fixed control draws, including legitimate repeated single-head draws.
    expected = make_groups(manifest["protocol"], manifest["arguments"]["random_draws"])
    actual = {name: [tuple(pair) for pair in pairs] for name, pairs in manifest["groups"].items()}
    if expected != actual:
        raise ValueError("Saved groups differ from the frozen control sampler")
    by_id = {(row["example_id"], row["marker_mode"]): row for row in examples}
    expected_ids = {f"{variant}:{manifest['resolved_seed_base'] + index * 100000 + offset}"
                    for index, variant in enumerate(VARIANTS) for offset in range(manifest["n_per_variant"])}
    if {row["example_id"] for row in examples} != expected_ids:
        raise ValueError("Example identities differ from the reserved seed schedule")
    for row in examples:
        if row["example_id"] != f"{row['variant']}:{row['seed']}":
            raise ValueError("Example identity/seed mismatch")
        clean, corrupt = row["clean_prompt"], row["corrupt_prompt"]
        if clean.count(row["clean_gold"]) != 1 or corrupt.count(row["corrupt_gold"]) != 1:
            raise ValueError("Ambiguous answer source span")
        if clean.replace(row["clean_gold"], "<ANSWER>") != corrupt.replace(row["corrupt_gold"], "<ANSWER>"):
            raise ValueError("Corruption differs outside the answer")
        if seeded_decoy(row["seed"]) not in clean or seeded_decoy(row["seed"]) not in corrupt:
            raise ValueError("Expected unrelated distractor is absent")
        if row["marker_mode"] == "marked":
            unmarked = by_id[row["example_id"], "unmarked"]
            for label in ("clean", "corrupt"):
                stripped = row[f"{label}_prompt"].replace(NEEDLE_START + " ", "").replace(" " + NEEDLE_END, "")
                if stripped != unmarked[f"{label}_prompt"]:
                    raise ValueError("Marker conditions differ beyond marker removal")
            for field in ("clean_gold", "corrupt_gold", "seed", "variant", "filler_words", "answer_tokens"):
                if row[field] != unmarked[field]:
                    raise ValueError(f"Marker pairing mismatch: {field}")


def diagnose(directory, pilot_directory):
    manifest, rows, examples, lookup, paired, self_error = load_checked(directory)
    pilot_manifest, pilot_rows, pilot_examples, pilot_lookup, _, pilot_error = load_checked(pilot_directory)
    if manifest["arguments"]["stage"] != "confirmation" or pilot_manifest["arguments"]["stage"] != "pilot":
        raise ValueError("Expected a confirmation and a pilot, respectively")
    if any(m["arguments"]["prompt_format"] != "chat" for m in (manifest, pilot_manifest)):
        raise ValueError("This diagnostic comparison requires two chat runs")
    validate_design(manifest, examples)
    validate_design(pilot_manifest, pilot_examples)
    if paired != json.loads((directory / "paired_analysis.json").read_text()):
        raise ValueError("Saved paired analysis does not reproduce")
    identities = {row["example_id"] for row in examples}
    if identities & {row["example_id"] for row in pilot_examples}:
        raise ValueError("Confirmation overlaps pilot identities")
    same_fields = ("source_sha256", "model_revision", "tokenizer_and_chat_template_sha256",
                   "packages", "dtype", "runtime_config", "protocol")
    for field in same_fields:
        if manifest[field] != pilot_manifest[field]:
            raise ValueError(f"Pilot and confirmation differ in {field}")

    focal = list(manifest["protocol"]["groups"])
    focal_conditions = {f"{action}:{name}" for name in focal for action in ("ablate", "patch")}
    selected_conditions = focal_conditions | {"clean", "corrupt"}
    result = {
        "audit": {
            "status": "passed", "rows": len(rows), "prompt_pair_cases": len(examples),
            "independent_identities": len(identities), "pilot_identity_overlap": 0,
            "historical_identity_overlap": 0, "saved_paired_analysis_matches": True,
            "self_patch_max_score_error": self_error, "pilot_self_patch_max_score_error": pilot_error,
            "same_as_pilot": list(same_fields),
            "controls_match_frozen_sampler": True, "answer_and_marker_only_prompt_edits": True,
        },
        "source_sha256": {f"{label}/{name}": sha256(path / name)
                          for label, path in (("confirmation", directory), ("chat_pilot", pilot_directory))
                          for name in ("manifest.json", "examples.jsonl", "rows.jsonl", "summary.json", "paired_analysis.json")},
        "interval_note": paired["method"],
        "diagnostic_note": "First-token decomposition and repeated-number content checks were defined after the pilot, before confirmation. Distractor-output classification, exact first-token switching counts, and per-control overlap inspection are additional post-confirmation diagnostics. Do not replace frozen exact-match metrics with extraction scores.",
        "comparison_note": "Pilot and confirmation have disjoint identities and different lengths/control counts. This is a descriptive replication comparison, not a paired length-effect estimate. Pointwise intervals are not multiplicity adjusted; zero bootstrap intervals for zero observed transfers do not establish a zero population rate.",
        "runtime": {
            "elapsed_hours_including_model_load": manifest["elapsed_seconds_including_model_load"] / 3600,
            "peak_allocated_cuda_gib": manifest["peak_cuda_memory_bytes"] / 1024 ** 3,
            "prompt_tokens_by_marker": {mode: [min(row["prompt_tokens"] for row in examples if row["marker_mode"] == mode),
                                               max(row["prompt_tokens"] for row in examples if row["marker_mode"] == mode)]
                                        for mode in ("marked", "unmarked")},
            "generation_stops": dict(Counter(row["generation_stop"] for row in rows)),
        },
        "conditions": [], "first_token_subgroups": [], "controls": [],
        "content_contrasts": [], "pilot_comparison": [], "changed_outputs": [],
    }
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["marker_mode"], row["condition"]].append(row)
    for (mode, condition), selected in sorted(buckets.items()):
        for variant in ["all"] + (sorted({row["variant"] for row in selected}) if condition in selected_conditions else []):
            subset = selected if variant == "all" else [row for row in selected if row["variant"] == variant]
            record = {"marker": mode, "condition": condition, "variant": variant,
                      **summarize_condition(subset, lookup, intervals=variant == "all")}
            result["conditions"].append(record)
        if condition in focal_conditions and condition.startswith("patch:"):
            for differ in (False, True):
                subset = [row for row in selected if row["first_answer_tokens_differ"] == differ]
                if subset:
                    result["first_token_subgroups"].append({"marker": mode, "condition": condition,
                        "first_answer_tokens_differ": differ, **summarize_condition(subset, lookup, intervals=True)})
        for row in selected:
            diag = row_diagnostics(row, lookup)
            if diag["changed_output"]:
                result["changed_outputs"].append({
                    **{key: row[key] for key in ("example_id", "variant", "marker_mode", "condition", "clean_gold",
                                                "corrupt_gold", "generated_text", "generation_stop", "answer_logprob_margin")},
                    "unrelated_decoy": seeded_decoy(row["seed"]),
                    "baseline_text": lookup[row["example_id"], mode, row["reference"]]["generated_text"], **diag,
                })

    address = {tuple(pair) for pair in manifest["groups"]["answer_address"]}
    for name in focal:
        controls = sorted(group for group in manifest["groups"] if group.startswith(f"random_{name}_"))
        distinct = {tuple(sorted(map(tuple, manifest["groups"][group]))) for group in controls}
        for mode in ("marked", "unmarked"):
            draws = []
            for group in controls:
                draws.append({"group": group, "heads": manifest["groups"][group],
                    "address_overlap": [list(pair) for pair in sorted(set(map(tuple, manifest["groups"][group])) & address)],
                    **{f"{action}_mean_delta_clean_logprob": statistics.fmean(row["delta_clean_logprob"] for row in buckets[mode, f"{action}:{group}"])
                       for action in ("ablate", "patch")}})
            result["controls"].append({"marker": mode, "focal_group": name, "draw_count": len(controls),
                "distinct_head_sets": len(distinct), "draws": draws,
                "mean_patch_delta": statistics.fmean(draw["patch_mean_delta_clean_logprob"] for draw in draws)})
            for reference, conditions in (("clean", ["clean"]), ("mean_matched_controls", [f"ablate:{group}" for group in controls])):
                values, variants = [], []
                for row in buckets[mode, f"ablate:{name}"]:
                    control_values = [row_diagnostics(lookup[row["example_id"], mode, condition], lookup)["clean_number_content"] for condition in conditions]
                    values.append(row_diagnostics(row, lookup)["clean_number_content"] - statistics.fmean(control_values))
                    variants.append(row["variant"])
                result["content_contrasts"].append({"marker": mode, "focal_group": name,
                    "reference": reference, "metric": "clean_number_content", **paired_interval(values, variants)})

    for mode in ("marked", "unmarked"):
        for condition in sorted(selected_conditions):
            pilot = [row for row in pilot_rows if row["marker_mode"] == mode and row["condition"] == condition]
            result["pilot_comparison"].append({"marker": mode, "condition": condition,
                "chat_4k_pilot": summarize_condition(pilot, pilot_lookup),
                "chat_8k_confirmation": summarize_condition(buckets[mode, condition], lookup)})
    patches = [row for row in rows if row["condition"].startswith("patch:")]
    result["all_patches"] = {
        "condition_rows_not_independent_trials": len(patches),
        "exact_donor_outputs": sum(row["exact_clean"] for row in patches),
        "sole_number_donor_outputs": sum(row["sole_number_is_clean"] for row in patches),
        "donor_content_diagnostic_outputs": sum(row_diagnostics(row, lookup)["clean_number_content"] for row in patches),
        "maximum_full_donor_minus_recipient_margin": max(row["answer_logprob_margin"] for row in patches),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts_publication/confirmation_8192_chat")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "artifacts_publication/pilot_4096_chat")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_publication/confirmation_8192_chat_diagnostics.json")
    args = parser.parse_args()
    result = diagnose(args.run_dir, args.pilot_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audit passed: {result['audit']['rows']} rows, {result['audit']['independent_identities']} identities.")
    print(f"Wrote condition, token, control, content, and changed-output diagnostics: {args.out}")


if __name__ == "__main__":
    main()
