#!/usr/bin/env python
"""Audit temporal runs and distinguish score, prefix, and complete-answer effects."""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.temporal import CORRUPTIONS, SCOPES, build_examples, prior_identities
from scripts.analyze_publication_confirmation import paired_interval
from scripts.analyze_publication_temporal import analyze, read_rows
from scripts.compare_publication_pilots import load_checked, sha256
from scripts.run_publication_confirmation import tokenizer_fingerprint, write_json


def donor_prefix(row, length):
    target = row["candidate_token_ids"]["clean"]
    if not 1 <= length <= len(target):
        raise ValueError("Prefix length must be within the candidate answer")
    return int(row["generated_token_ids"][:length] == target[:length])


def summarize_condition(rows, lookup):
    if not rows or len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("Summarize one condition and corruption per independent identity")
    baselines = [lookup[row["case_id"], row["reference"]] for row in rows]
    deltas = [[a - b for a, b in zip(row["candidate_scores"]["clean"]["token_logprobs"],
                                    base["candidate_scores"]["clean"]["token_logprobs"])]
              for row, base in zip(rows, baselines)]
    probabilities = {label: [math.exp(row["candidate_scores"][label]["token_logprobs"][row["shared_prefix_tokens"]])
                             for row in rows] for label in ("clean", "corrupt")}
    result = {"n_identities": len(rows),
              "mean_clean_token_logprob_delta": [statistics.fmean(d[i] for d in deltas) for i in range(6)],
              "count_donor_prefix_at_least": [sum(donor_prefix(row, i) for row in rows) for i in range(1, 7)],
              "count_donor_prefix_at_least_and_exact_decoy": [sum(donor_prefix(row, i) and row["exact_decoy"] for row in rows)
                                                              for i in range(1, 7)],
              "count_changed_from_reference": sum(row["generated_token_ids"] != base["generated_token_ids"]
                                                  for row, base in zip(rows, baselines))}
    for key in ("delta_clean_logprob", "answer_logprob_margin", "clean_decoy_logprob_margin",
                "aligned_suffix_fraction_clean"):
        result[f"mean_{key}"] = statistics.fmean(row[key] for row in rows)
    for key in ("exact_clean", "exact_corrupt", "exact_decoy", "sole_number_is_clean", "clean_number_content",
                "six_digits_only", "first_differing_token_is_clean", "aligned_suffix_exact_clean"):
        result[f"count_{key}"] = sum(row[key] for row in rows)
    for label, values in probabilities.items():
        result[f"mean_first_differing_token_probability_{label}"] = statistics.fmean(values)
        result[f"maximum_first_differing_token_probability_{label}"] = max(values)
    result["maximum_answer_logprob_margin"] = max(row["answer_logprob_margin"] for row in rows)
    result["count_positive_clean_decoy_margin"] = sum(row["clean_decoy_logprob_margin"] > 0 for row in rows)
    result["count_other_six_digit_answer"] = sum(row["six_digits_only"] and not any(row[f"exact_{label}"]
                                                 for label in ("clean", "corrupt", "decoy")) for row in rows)
    result["count_not_six_digits_only"] = sum(not row["six_digits_only"] for row in rows)
    return result


def load_temporal(directory, stage):
    result = analyze(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["arguments"]["stage"] != stage:
        raise ValueError(f"Expected temporal stage {stage}")
    for name, value in result.items():
        if json.loads((directory / f"{name}.json").read_text()) != value:
            raise ValueError(f"Saved temporal {name} does not reproduce")
    return manifest, read_rows(directory / "rows.jsonl"), read_rows(directory / "examples.jsonl"), result


def diagnose(directory, smoke_directory, earlier_directory, verify_tokenizer=False):
    manifest, rows, examples, analysis = load_temporal(directory, "confirmation")
    smoke, smoke_rows, smoke_examples, smoke_analysis = load_temporal(smoke_directory, "smoke")
    same_fields = ("protocol", "original_protocol", "source_sha256", "packages", "dtype", "runtime_config",
                   "groups", "model_revision", "tokenizer_and_chat_template_sha256")
    for field in same_fields:
        if manifest[field] != smoke[field]:
            raise ValueError(f"Smoke and confirmation differ in {field}")
    if manifest["arguments"]["chunk_size"] != smoke["arguments"]["chunk_size"]:
        raise ValueError("Smoke and confirmation chunk sizes differ")
    gate = manifest["smoke_gate"]
    if gate["rows_sha256"] != smoke["rows_sha256"] or gate["validation"] != smoke_analysis["validation"]:
        raise ValueError("Confirmation did not use the supplied smoke")
    prior = prior_identities(ROOT, manifest["protocol"]["protocol"])
    if any((ex["variant"], ex["seed"]) in prior for ex in examples + smoke_examples):
        raise ValueError("Temporal identities overlap earlier data")
    if {ex["example_id"] for ex in examples} & {ex["example_id"] for ex in smoke_examples}:
        raise ValueError("Temporal smoke and confirmation identities overlap")

    earlier, earlier_rows, _, _, earlier_paired, _ = load_checked(earlier_directory)
    if earlier_paired != json.loads((earlier_directory / "paired_analysis.json").read_text()):
        raise ValueError("Earlier confirmation paired analysis does not reproduce")
    for field in ("model_revision", "tokenizer_and_chat_template_sha256", "packages", "dtype"):
        if manifest[field] != earlier[field]:
            raise ValueError(f"Temporal and earlier confirmation differ in {field}")
    if manifest["original_protocol"] != earlier["protocol"]:
        raise ValueError("Temporal focal groups were not inherited from the earlier protocol")

    lookup = {(row["case_id"], row["condition"]): row for row in rows}
    focal = manifest["protocol"]["focal_groups"]
    buckets = defaultdict(list)
    for row in rows:
        if not row["condition"].startswith("self_patch_"):
            buckets[row["corruption"], row["condition"]].append(row)
    result = {
        "audit": {"status": "passed", "confirmation": analysis["validation"], "smoke": smoke_analysis["validation"],
                  "all_saved_analyses_reproduce": True, "same_smoke_fields": list(same_fields),
                  "smoke_gate_matches": True, "prior_identity_count": len(prior), "prior_identity_overlap": 0,
                  "smoke_confirmation_identity_overlap": 0, "earlier_confirmation_reaudited": True},
        "source_sha256": {f"{label}/{path.name}": sha256(path)
                          for label, folder in (("temporal", directory), ("smoke", smoke_directory), ("earlier", earlier_directory))
                          for path in sorted(folder.iterdir()) if path.suffix in (".json", ".jsonl")},
        "analysis_source_sha256": sha256(Path(__file__)),
        "notes": ["Primary inference uses 40 identities, not 80 corruption cases or 3520 condition rows. Smoke stays separate.",
                  "Scope and control contrasts are prespecified; prefix lengths, output transitions, probability diagnostics, and baseline-success sensitivity are post-run diagnostics.",
                  "Teacher-forced later-token probabilities condition on the candidate prefix, not the greedy trajectory. At the first differing digit, both candidate prefixes agree.",
                  "A shared first digit is not transferred content. Prefix matches and suffix fractions are not full-answer transfer.",
                  "A prefix can match the donor and the unrelated distractor simultaneously; these matches do not uniquely identify the source of generated content.",
                  "Bootstrap intervals are pointwise, conditional on fixed control draws, and not multiplicity-adjusted. No events does not establish a zero population rate.",
                  "Earlier confirmation and temporal identities/corruptions differ; their comparison is descriptive, not a paired estimate of a protocol effect."],
        "runtime": {}, "conditions": [], "smoke_conditions": [], "scope_transitions": [],
        "prefix_scope_contrasts": [], "baseline_failures": [], "baseline_success_sensitivity": [],
        "changed_outputs": [], "earlier_comparison": [],
        "primary_contrasts": [item for item in analysis["paired_analysis"]["contrasts"]
                              if item["metric"] in ("delta_clean_logprob", "delta_exact_clean")],
    }
    for label, meta, rr, ee in (("confirmation", manifest, rows, examples), ("smoke", smoke, smoke_rows, smoke_examples)):
        result["runtime"][label] = {
            "elapsed_minutes_total": meta["elapsed_seconds_total"] / 60,
            "peak_allocated_cuda_gib": meta["peak_cuda_memory_bytes"] / 1024 ** 3,
            "prompt_tokens": [min(ex["prompt_tokens"] for ex in ee), max(ex["prompt_tokens"] for ex in ee)],
            "generation_stops": dict(Counter(row["generation_stop"] for row in rr)),
        }
    for (mode, condition), selected in sorted(buckets.items()):
        variants = ["all"] + (sorted({row["variant"] for row in selected})
                               if condition in ("clean", "corrupt") or condition.split(":")[-1] in focal else [])
        for variant in variants:
            subset = selected if variant == "all" else [row for row in selected if row["variant"] == variant]
            result["conditions"].append({"corruption": mode, "condition": condition, "variant": variant,
                                         **summarize_condition(subset, lookup)})
        for row in selected:
            baseline = lookup[row["case_id"], row["reference"]]
            if row["generated_token_ids"] != baseline["generated_token_ids"] or (
                    condition in ("clean", "corrupt") and not row[f"exact_{condition}"]):
                record = {key: row[key] for key in ("case_id", "example_id", "variant", "corruption", "condition",
                          "clean_gold", "corrupt_gold", "unrelated_decoy", "generated_text", "generation_stop",
                          "exact_clean", "exact_corrupt", "exact_decoy", "answer_logprob_margin", "clean_decoy_logprob_margin")}
                record["reference_text"] = baseline["generated_text"]
                result["baseline_failures" if condition in ("clean", "corrupt") else "changed_outputs"].append(record)
    smoke_lookup = {(row["case_id"], row["condition"]): row for row in smoke_rows}
    for mode in CORRUPTIONS:
        for group in focal:
            for scope in SCOPES:
                condition = f"patch:{scope}:{group}"
                result["smoke_conditions"].append({"corruption": mode, "condition": condition,
                    **summarize_condition([r for r in smoke_rows if r["corruption"] == mode and r["condition"] == condition], smoke_lookup)})
                selected = buckets[mode, condition]
                successful = [r for r in selected if lookup[r["case_id"], "clean"]["exact_clean"]
                              and lookup[r["case_id"], "corrupt"]["exact_corrupt"]]
                result["baseline_success_sensitivity"].append({"corruption": mode, "condition": condition,
                    **summarize_condition(successful, lookup)})
            for left, right in ((f"patch:decode_only:{group}", "corrupt"),
                                (f"patch:prompt_and_decode:{group}", f"patch:prompt_only:{group}")):
                selected = buckets[mode, left]
                changed = [{"case_id": r["case_id"], "before": lookup[r["case_id"], right]["generated_text"],
                            "after": r["generated_text"]} for r in selected
                           if r["generated_token_ids"] != lookup[r["case_id"], right]["generated_token_ids"]]
                result["scope_transitions"].append({"corruption": mode, "group": group, "left": left, "right": right,
                                                    "n_identities": len(selected), "n_changed": len(changed), "changes": changed})
            if mode == "different_first":
                selected = buckets[mode, f"patch:prompt_and_decode:{group}"]
                for length in (1, 2, 3, 6):
                    values = [donor_prefix(r, length) - donor_prefix(lookup[r["case_id"], f"patch:prompt_only:{group}"], length)
                              for r in selected]
                    result["prefix_scope_contrasts"].append({"group": group, "prefix_length": length,
                        "contrast": "different_first:prompt_and_decode_minus_prompt_only",
                        **paired_interval(values, [r["variant"] for r in selected])})
            old = [r for r in earlier_rows if r["marker_mode"] == "unmarked" and r["condition"] == f"patch:{group}"
                   and r["first_answer_tokens_differ"] == (mode == "different_first")]
            selected = buckets[mode, f"patch:prompt_only:{group}"]
            result["earlier_comparison"].append({"corruption": mode, "group": group,
                "earlier_n": len(old), "temporal_n": len(selected),
                "earlier_mean_delta_clean_logprob": statistics.fmean(r["delta_clean_logprob"] for r in old),
                "temporal_mean_delta_clean_logprob": statistics.fmean(r["delta_clean_logprob"] for r in selected),
                "earlier_exact_donor": sum(r["exact_clean"] for r in old),
                "temporal_exact_donor": sum(r["exact_clean"] for r in selected)})
    patches = [row for row in rows if row["condition"].startswith("patch:")]
    result["all_patches"] = {"condition_rows_not_independent_trials": len(patches),
        **{f"count_{key}": sum(r[key] for r in patches) for key in ("exact_clean", "sole_number_is_clean", "clean_number_content", "exact_decoy")},
        "maximum_answer_logprob_margin": max(r["answer_logprob_margin"] for r in patches),
        "control_outputs_changed_from_corrupt": sum(not r["exact_corrupt"] for r in patches if ":random_" in r["condition"])}
    if verify_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["protocol"]["model_id"], revision=manifest["model_revision"],
                                                  local_files_only=True, trust_remote_code=False)
        if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_and_chat_template_sha256"]:
            raise ValueError("Local tokenizer fingerprint differs")
        for stage, ee, rr in (("confirmation", examples, rows), ("smoke", smoke_examples, smoke_rows)):
            if build_examples(tokenizer, manifest["protocol"], stage, prior) != ee:
                raise ValueError("Full example regeneration differs")
            if any(tokenizer.decode(r["generated_token_ids"], skip_special_tokens=True) != r["generated_text"] for r in rr):
                raise ValueError("Generated text does not match saved token IDs")
        result["audit"]["tokenizer_only_check"] = {"status": "passed", "regenerated_cases": len(examples) + len(smoke_examples),
                                                   "redecoded_generations": len(rows) + len(smoke_rows)}
    return result


def render_figure(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), layout="constrained")
        for ax, mode, title in zip(axes[:2], CORRUPTIONS, ("A. Different first digit", "B. Shared first digit")):
            for group, label, color, offset in (("address_L22H7", "L22H7", "#0072B2", -.07),
                                                ("answer_address", "Three-head group", "#D55E00", .07)):
                values = [next(c for c in result["primary_contrasts"] if c["metric"] == "delta_clean_logprob"
                               and c["contrast"] == f"{mode}:{scope}:{group}_minus_corrupt") for scope in SCOPES]
                means = [c["mean"] for c in values]
                errors = [[c["mean"] - c["ci_low"] for c in values], [c["ci_high"] - c["mean"] for c in values]]
                ax.errorbar([i + offset for i in range(3)], means, yerr=errors, marker="o", capsize=3,
                            color=color, label=label, linewidth=1.5)
            ax.axhline(0, color="0.7", linewidth=.8)
            ax.set(xticks=range(3), xticklabels=["Prompt\nonly", "Decode\nonly", "Prompt +\ndecode"],
                   title=title, ylabel="Mean donor-token log-probability change", ylim=(-.18, 2.4))
            ax.legend(frameon=False, loc="upper left", fontsize=9)
        ax = axes[2]
        for scope, label, color in zip(SCOPES, ("Prompt only", "Decode only", "Prompt + decode"), ("#0072B2", "#777777", "#D55E00")):
            record = next(c for c in result["conditions"] if c["variant"] == "all" and c["corruption"] == "different_first"
                          and c["condition"] == f"patch:{scope}:answer_address")
            ax.plot(range(1, 7), record["count_donor_prefix_at_least"], "o-", label=label, color=color, linewidth=1.5)
        ax.set(title="C. Prefix match, three-head group", xlabel="Consecutive donor digits from answer start",
               ylabel="Identities (out of 40)", xticks=range(1, 7), yticks=range(0, 41, 10), ylim=(-2, 41))
        ax.legend(frameon=False, fontsize=9)
        fig.suptitle("Temporal patching increases donor scores, but not complete answer transfer", fontsize=14)
        fig.supxlabel("40 paired identities; unmarked 8K chat. A/B: 95% pointwise identity-bootstrap intervals.\n"
                      "C: different-first-digit cases only; prefix matches are not complete answers.", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"temporal_scope_comparison.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts_publication/temporal_confirmation_8192_chat")
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "artifacts_publication/temporal_smoke_8192_chat")
    parser.add_argument("--earlier-dir", type=Path, default=ROOT / "artifacts_publication/confirmation_8192_chat")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_publication/temporal_confirmation_8192_chat_diagnostics.json")
    parser.add_argument("--verify-tokenizer", action="store_true", help="Regenerate examples and decode outputs using the cached pinned tokenizer, without weights")
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run_dir, args.smoke_dir, args.earlier_dir, args.verify_tokenizer)
    write_json(args.out, result)
    if args.figure_dir:
        render_figure(result, args.figure_dir)
    print(json.dumps({"audit": result["audit"], "all_patches": result["all_patches"]}, indent=2))
    print(f"Wrote temporal diagnostics: {args.out}")


if __name__ == "__main__":
    main()
