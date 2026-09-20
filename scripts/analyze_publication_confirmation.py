#!/usr/bin/env python
"""Paired, variant-stratified summaries of one completed confirmation run."""

import argparse
import json
from pathlib import Path

import numpy as np


def paired_interval(values, variants, draws=4000, seed=730022):
    if len(values) != len(variants) or not values or draws < 1:
        raise ValueError("Need aligned, nonempty paired observations and positive bootstrap draws")
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite paired effect")
    rng = np.random.default_rng(seed)
    resampled = np.zeros(draws)
    for variant in sorted(set(variants)):
        subset = data[np.asarray(variants) == variant]
        resampled += rng.choice(subset, size=(draws, len(subset)), replace=True).sum(axis=1) / len(data)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {"n_identities": len(data), "mean": float(data.mean()), "ci_low": float(low), "ci_high": float(high)}


def analyze(directory, draws):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only analyze completed runs; retain failed/partial output for debugging")
    with (directory / "rows.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    lookup = {(row["example_id"], row["marker_mode"], row["condition"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate example/marker/condition row")
    identities = sorted({row["example_id"] for row in rows})
    conditions = {"clean", "corrupt", "self_patch_clean", "self_patch_corrupt"}
    conditions.update(f"{action}:{name}" for name in manifest["groups"] for action in ("ablate", "patch"))
    expected = {(identity, mode, condition) for identity in identities for mode in ("marked", "unmarked") for condition in conditions}
    if set(lookup) != expected or len(identities) != manifest["n_independent_example_ids"] or len(rows) != manifest["rows_written"]:
        raise ValueError("Incomplete or unexpected condition grid")
    variants = [lookup[(identity, "marked", "clean")]["variant"] for identity in identities]
    contrasts = []

    def contrast(label, metric, left_mode, left_conditions, right_mode, right_conditions):
        values = []
        for identity in identities:
            left = np.mean([lookup[(identity, left_mode, condition)][metric] for condition in left_conditions])
            right = np.mean([lookup[(identity, right_mode, condition)][metric] for condition in right_conditions])
            values.append(float(left - right))
        contrasts.append({"contrast": label, "metric": metric, **paired_interval(values, variants, draws)})

    for mode in ("marked", "unmarked"):
        for metric in ("delta_clean_logprob", "delta_answer_margin", "delta_exact_clean"):
            contrast(f"{mode}:patch_L22H7_minus_L20H7", metric, mode, ["patch:address_L22H7"], mode, ["patch:support_L20H7"])
            contrast(f"{mode}:ablate_L20H7_minus_L22H7", metric, mode, ["ablate:support_L20H7"], mode, ["ablate:address_L22H7"])
            for name in manifest["protocol"]["groups"]:
                controls = sorted(group for group in manifest["groups"] if group.startswith(f"random_{name}_"))
                if not controls:
                    raise ValueError(f"Missing matched controls for {name}")
                for action in ("ablate", "patch"):
                    contrast(f"{mode}:{action}:{name}_minus_mean_matched_controls", metric, mode,
                             [f"{action}:{name}"], mode, [f"{action}:{control}" for control in controls])
        for condition in sorted(conditions):
            if condition.startswith(("ablate:", "patch:")):
                baseline = "clean" if condition.startswith("ablate:") else "corrupt"
                contrast(f"{mode}:{condition}_minus_{baseline}", "clean_mean_logprob", mode, [condition], mode, [baseline])

    for condition in sorted(conditions - {"self_patch_clean", "self_patch_corrupt"}):
        for metric in ("delta_clean_logprob", "delta_exact_clean", "exact_clean"):
            contrast(f"unmarked_minus_marked:{condition}", metric, "unmarked", [condition], "marked", [condition])
    return {
        "protocol": manifest["protocol"]["protocol"],
        "stage": manifest["arguments"]["stage"], "n_independent_example_ids": len(identities),
        "method": "Paired percentile bootstrap, resampling identities within each variant; 95% pointwise intervals. Random draws are averaged within identity, not counted as extra examples. Intervals condition on the fixed sampled control sets.",
        "warning": "Pilot intervals are descriptive only. No selection or multiplicity adjustment; no complete-circuit inference. Do not pool raw/chat or length/position repetitions as independent observations.",
        "bootstrap_draws": draws, "contrasts": contrasts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.bootstrap_draws)
    path = args.run_dir / "paired_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(result['contrasts'])} paired descriptive contrasts: {path}")


if __name__ == "__main__":
    main()
