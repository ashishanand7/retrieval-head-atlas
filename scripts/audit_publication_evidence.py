#!/usr/bin/env python
"""Recompute the publication's core claims from historical row-level artifacts."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows, group):
    selected = [row for row in rows if row["group"] == group]
    result = {(row["variant"], int(row["seed"])): row for row in selected}
    if not selected or len(result) != len(selected):
        raise ValueError(f"Missing or duplicate example rows for {group}")
    return result


def average(rows, field):
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Missing or non-finite values: {field}")
    return statistics.fmean(values)


def audit(root=ROOT):
    artifacts = root / "artifacts_phase2"
    source_hashes = {}

    def read(name):
        path = artifacts / name
        source_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_rows(path)

    discovery = read("semantic_single_head_sweep_neighborhood61_8192_n8_query.csv")
    discovery_ids = {(row["variant"], int(row["seed"])) for row in discovery}
    settings, identities = [], []
    for target in (8192, 16384):
        for position in (0.1, 0.5, 0.9):
            suffix = "" if (target, position) == (8192, 0.1) else f"_frac{str(position).replace('.', 'p')}"
            single_rows = read(f"semantic_single_head_patching_{target}_n8{suffix}.csv")
            group_rows = read(f"semantic_component_patching_functional_{target}_n8{suffix}.csv")
            ablation_rows = read(f"semantic_group_ablation_functional_controls_{target}_n8{suffix}_query.csv")
            single = keyed(single_rows, "address_L22H7")
            address = keyed(group_rows, "answer_address")
            support = keyed(group_rows, "non_address_core")
            ablation = keyed(ablation_rows, "non_address_core")
            keys = set(single)
            if not keys == set(address) == set(support) == set(ablation):
                raise ValueError(f"Example alignment mismatch at {target}/{position}")
            identities.append(keys)
            for key in keys:
                row = single[key]
                comparisons = (
                    (float(row["patched_logprob"]) - float(row["corrupt_clean_logprob"]), float(row["delta_patch"])),
                    (float(row["clean_logprob"]) - float(row["corrupt_clean_logprob"]), float(row["recovery_denominator"])),
                    (float(row["clean_logprob"]), float(ablation[key]["baseline_logprob"])),
                    (float(row["clean_logprob"]), float(address[key]["clean_logprob"])),
                )
                if not all(math.isclose(a, b, abs_tol=1e-8) for a, b in comparisons):
                    raise ValueError(f"Score arithmetic or baseline mismatch: {target}/{position}/{key}")
            effect = average(single.values(), "delta_patch")
            group_effect = average(address.values(), "delta_patch")
            gap = average(single.values(), "recovery_denominator")
            settings.append({
                "target_tokens": target, "needle_frac": position,
                "n_examples": len(keys), "overlap_with_discovery": len(keys & discovery_ids),
                "l22h7_patch": effect, "address_patch": group_effect,
                "support_patch": average(support.values(), "delta_patch"),
                "support_ablation": average(ablation.values(), "delta_logprob"),
                "clean_corrupt_gap": gap,
                "l22h7_over_address_ratio_of_means": effect / group_effect,
                "l22h7_recovery_ratio_of_means": effect / gap,
                "l22h7_recovery_mean_of_ratios": average(single.values(), "recovery_fraction"),
                "prompt_tokens_min": min(int(row["prompt_tokens"]) for row in single.values()),
                "prompt_tokens_max": max(int(row["prompt_tokens"]) for row in single.values()),
            })

    interactions = read("semantic_patch_interaction_answer_address_with_ablate_only_8192_n8.csv")
    by_example = {}
    for row in interactions:
        key = row["variant"], int(row["seed"])
        conditions = by_example.setdefault(key, {})
        if row["condition"] in conditions:
            raise ValueError(f"Duplicate interaction row: {key}")
        conditions[row["condition"]] = row
    interaction_results = {}
    for group in ("non_address_core", "query_tail", "first_token_sink"):
        under_ablation, changes = [], []
        for conditions in by_example.values():
            combined = float(conditions[f"patch_plus_ablate_{group}"]["patched_logprob"])
            ablate = float(conditions[f"ablate_only_{group}"]["patched_logprob"])
            patch = float(conditions["patch_only"]["delta_patch"])
            under_ablation.append(combined - ablate)
            changes.append(combined - ablate - patch)
        interaction_results[group] = {
            "n_examples": len(under_ablation),
            "patch_effect_under_ablation": statistics.fmean(under_ablation),
            "change_from_patch_only": statistics.fmean(changes),
        }

    return {
        "scope": "Historical core patching/ablation arithmetic, identities, and interaction control; not a GPU rerun or exhaustive artifact audit.",
        "settings": settings,
        "unique_example_identities_across_settings": len(set.union(*identities)),
        "identities_shared_by_all_settings": len(set.intersection(*identities)),
        "interaction_control": interaction_results,
        "historical_interval_method": "Pointwise mean +/- 1.96 * sample_sd / sqrt(n); not a bootstrap or selection-adjusted interval.",
        "source_sha256": source_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
        print(f"Wrote evidence audit: {args.out}")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
