#!/usr/bin/env python
"""Audit multi-fact results and distinguish target switching from donor agreement."""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.multifact import CORRUPTIONS, SCOPES, VIEWS, build_examples, prior_ids
from scripts.analyze_publication_multifact import analyze, read_compressed_rows, read_rows
from scripts.compare_publication_pilots import sha256
from scripts.diagnose_publication_specificity import load_specificity
from scripts.run_publication_confirmation import tokenizer_fingerprint, write_json


def summarize_condition(rows, lookup):
    if not rows or len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("Summarize one observation per independent identity")
    if len({(r["view"], r["corruption"], r["condition"]) for r in rows}) != 1:
        raise ValueError("Do not pool conditions, distances or corruptions")
    bases = [lookup[r["case_id"], r["reference"]] for r in rows]
    donors = [lookup[r["case_id"], "clean"] for r in rows]
    recipients = [lookup[r["case_id"], "corrupt"] for r in rows]
    eligible = [r for r, d, c in zip(rows, donors, recipients) if d["exact_clean"] and c["exact_corrupt"]]
    token_delta = [statistics.fmean(r["candidate_scores"]["clean"]["token_logprobs"][i]
                                   - b["candidate_scores"]["clean"]["token_logprobs"][i]
                                   for r, b in zip(rows, bases)) for i in range(6)]
    total = sum(token_delta)
    result = {"n_identities": len(rows), "mean_clean_token_logprob_delta": token_delta,
        "first_digit_fraction_of_net_sequence_delta": token_delta[0] / total if abs(total) > 1e-12 else None,
        "count_new_target_from_reference": sum(r["exact_clean"] and not b["exact_clean"] for r, b in zip(rows, bases)),
        "count_target_lost_from_reference": sum(b["exact_clean"] and not r["exact_clean"] for r, b in zip(rows, bases)),
        "count_changed_token_sequence": sum(r["generated_token_ids"] != b["generated_token_ids"] for r, b in zip(rows, bases)),
        "count_matches_actual_donor_sequence": sum(r["generated_token_ids"] == d["generated_token_ids"] for r, d in zip(rows, donors)),
        "count_new_actual_donor_sequence": sum(r["generated_token_ids"] == d["generated_token_ids"]
                                                and b["generated_token_ids"] != d["generated_token_ids"]
                                                for r, b, d in zip(rows, bases, donors)),
        "both_baselines_correct": {"n": len(eligible), "count_exact_target": sum(r["exact_clean"] for r in eligible),
                                   "identities": [r["example_id"] for r in eligible]},
        "rank_clean_but_not_exact": sum(r["clean_beats_all_candidates"] and not r["exact_clean"] for r in rows),
        "exact_clean_but_not_ranked_first": sum(r["exact_clean"] and not r["clean_beats_all_candidates"] for r in rows),
        "generation_stops": dict(Counter(r["generation_stop"] for r in rows)),
        "mean_generated_tokens": statistics.fmean(len(r["generated_token_ids"]) for r in rows)}
    for key in ("delta_clean_logprob", "delta_best_other_margin", "clean_best_other_margin", "answer_logprob_margin"):
        result[f"mean_{key}"] = statistics.fmean(r[key] for r in rows)
    for key in ("exact_clean", "exact_corrupt", "exact_other_fact", "clean_number_content", "first_number_is_clean",
                "contains_clean_number", "clean_beats_all_candidates", "six_digits_only", "verbatim_clean",
                "exact_resample", "resample_number_content", "contains_reference_number"):
        result[f"count_{key}"] = sum(r[key] for r in rows)
    return result


def load_run(directory, stage):
    analysis = analyze(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["arguments"]["stage"] != stage:
        raise ValueError("Wrong multi-fact run stage")
    for name, value in analysis.items():
        if json.loads((directory / f"{name}.json").read_text()) != value:
            raise ValueError(f"Saved multi-fact {name} does not reproduce")
    return (manifest, read_compressed_rows(directory, manifest), read_rows(directory / "examples.jsonl"),
            read_rows(directory / "references.jsonl"), analysis)


def diagnose(directory, smoke_directory, earlier_directory, verify_tokenizer=False):
    manifest, rows, examples, references, analysis = load_run(directory, "confirmation")
    smoke, smoke_rows, smoke_examples, smoke_refs, smoke_analysis = load_run(smoke_directory, "smoke")
    for key in ("protocol", "original_protocol", "source_sha256", "groups", "packages", "dtype", "runtime_config",
                "model_revision", "tokenizer_and_chat_template_sha256"):
        if manifest[key] != smoke[key]:
            raise ValueError(f"Smoke and confirmation differ: {key}")
    if manifest["arguments"]["chunk_size"] != smoke["arguments"]["chunk_size"]:
        raise ValueError("Smoke and confirmation chunk sizes differ")
    for key in ("rows_sha256", "references_sha256"):
        if manifest["smoke_gate"][key] != smoke[key]:
            raise ValueError("Supplied smoke differs from confirmation gate")
    if manifest["smoke_gate"]["validation"] != smoke_analysis["validation"]:
        raise ValueError("Smoke gate validation differs")
    prior = prior_ids(ROOT, manifest["protocol"]["protocol"])
    schedules = [{r["seed"] for r in records} for records in (examples, references, smoke_examples, smoke_refs)]
    if sum(map(len, schedules)) != len(set.union(*schedules)) or set.union(*schedules) & {seed for _, seed in prior}:
        raise ValueError("Study/reference/stage or prior seed overlap")
    earlier, old_rows, old_examples, _, old_analysis = load_specificity(earlier_directory, "confirmation")
    for key in ("original_protocol", "groups", "packages", "dtype", "model_revision", "tokenizer_and_chat_template_sha256"):
        if manifest[key] != earlier[key]:
            raise ValueError(f"Earlier specificity configuration differs: {key}")
    result = {
        "audit": {"status": "passed", "confirmation": analysis["validation"], "smoke": smoke_analysis["validation"],
                  "all_saved_analyses_reproduce": True, "smoke_gate_matches": True,
                  "both_compressed_and_uncompressed_hashes_verified": True,
                  "prior_identity_count": len(prior), "prior_identity_overlap": 0, "study_reference_stage_overlap": 0,
                  "earlier_specificity_reaudited": old_analysis["validation"]["status"]},
        "source_sha256": {f"{label}/{path.name}": sha256(path)
                          for label, folder in (("confirmation", directory), ("smoke", smoke_directory), ("specificity", earlier_directory))
                          for path in sorted(folder.iterdir()) if path.suffix in (".json", ".jsonl", ".gz")},
        "analysis_source_sha256": sha256(Path(__file__)),
        "notes": ["Confirmation N=40; smoke, repeated conditions and reference donors are not pooled as independent trials.",
                  "Frozen contrasts are retained; token decomposition, output transitions, position breakdown and baseline-success sensitivity are post-run diagnostics.",
                  "exact_clean measures the known target code, not necessarily the actual donor output when that donor is wrong. Existing recipient target answers are not induced transfers.",
                  "Successful patching redirects a recipient away from its own correct answer; it is not an increase in ordinary task accuracy.",
                  "All four candidates remain in both prompts. Value swap changes two code spans; query selection changes a name. They do not match corruption size or isolate a single computation.",
                  "Earlier versus current comparisons change prose, fact inventory, templates and corruption together; candidate availability is a hypothesis, not an isolated causal factor.",
                  "Later teacher-forced scores condition on supplied candidate prefixes. First-digit dominance need not imply an exclusively first-digit function.",
                  "Intervals are pointwise and conditional on five fixed control draws. Secondary contrasts are descriptive, not multiplicity-adjusted.",
                  "Positive transfer calibrates actual donor reproduction, not correctness, sparse sufficiency or necessity. Token-budget stops remain censored."],
        "conditions": [], "smoke_conditions": [], "positive_control": {}, "patch_totals": {}, "runtime": {},
        "baseline_failures": [], "changed_focal_outputs": [], "budget_outputs": [], "scope_transitions": [],
        "scope_repair_examples": [], "common_baseline_success_sensitivity": [], "target_position_breakdown": [],
        "earlier_comparison": [], "frozen_contrasts": analysis["paired_analysis"]["contrasts"],
    }
    focal = set(manifest["protocol"]["distance_groups"] + manifest["protocol"]["ablation_groups"])
    for stage, meta, rr, ee, audits in (("confirmation", manifest, rows, examples, analysis),
                                       ("smoke", smoke, smoke_rows, smoke_examples, smoke_analysis)):
        lookup = {(r["case_id"], r["condition"]): r for r in rr}
        buckets = defaultdict(list)
        for row in rr:
            if not row["condition"].startswith("self_patch_"):
                buckets[row["view"], row["corruption"], row["condition"]].append(row)
        for (view, mode, condition), selected in sorted(buckets.items()):
            families = ["all"]
            if stage == "confirmation" and (condition in ("clean", "corrupt", "positive_transfer") or condition.split(":")[-1] in focal):
                families += sorted({r["variant"] for r in selected})
            for family in families:
                subset = selected if family == "all" else [r for r in selected if r["variant"] == family]
                result["conditions" if stage == "confirmation" else "smoke_conditions"].append({
                    "view": view, "corruption": mode, "condition": condition, "family": family,
                    **summarize_condition(subset, lookup)})
        positive = [r for r in rr if r["condition"] == "positive_transfer"]
        result["positive_control"][stage] = {
            "cases_not_independent_identities": len(positive),
            "token_sequences_equal_actual_donor": sum(r["generated_token_ids"] == lookup[r["case_id"], "clean"]["generated_token_ids"] for r in positive),
            "stopping_conditions_equal_actual_donor": sum(r["generation_stop"] == lookup[r["case_id"], "clean"]["generation_stop"] for r in positive),
            "correct_target_outputs": sum(r["exact_clean"] for r in positive),
            "incorrect_donor_outputs_reproduced": sum(not r["exact_clean"] for r in positive),
            "recorded_branch_step_logit_checks": sum(len(r["generation_steps"]) + sum(len(s["steps"]) for s in r["candidate_scores"].values()) for r in positive),
            "max_logit_error": audits["validation"]["positive_max_logit_error"],
            "max_token_score_error": audits["validation"]["positive_max_token_score_error"]}
        patches = [r for r in rr if r["condition"].startswith("patch:")]
        controls = [r for r in patches if ":random_" in r["condition"]]
        result["patch_totals"][stage] = {"rows_not_independent_trials": len(patches), "control_rows": len(controls),
            "control_exact_target_outputs": sum(r["exact_clean"] for r in controls),
            "control_new_target_outputs": sum(r["exact_clean"] and not lookup[r["case_id"], "corrupt"]["exact_clean"] for r in controls),
            "focal_new_target_outputs": sum(r["exact_clean"] and not lookup[r["case_id"], "corrupt"]["exact_clean"] for r in patches if ":random_" not in r["condition"])}
        result["runtime"][stage] = {"elapsed_minutes": meta["elapsed_seconds_total"] / 60,
            "peak_allocated_cuda_gib": meta["peak_cuda_memory_bytes"] / 1024 ** 3,
            "generation_stops": dict(Counter(r["generation_stop"] for r in rr)),
            "prompt_tokens": [min(e["prompt_tokens"] for e in ee), max(e["prompt_tokens"] for e in ee)],
            "source_to_question_tokens": {view: [min(e["source_spans"][code]["distance_to_question"] for e in ee if e["view"] == view for code in e["codes"]),
                                                  max(e["source_spans"][code]["distance_to_question"] for e in ee if e["view"] == view for code in e["codes"])] for view in VIEWS}}
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    for row in rows:
        baseline = lookup[row["case_id"], row["reference"]]
        failed_baseline = row["condition"] in ("clean", "corrupt") and not row[f"exact_{row['condition']}"]
        changed = row["condition"].split(":")[-1] in focal and row["generated_token_ids"] != baseline["generated_token_ids"]
        if failed_baseline or changed or row["generation_stop"] == "token_budget":
            record = {key: row[key] for key in ("case_id", "example_id", "variant", "view", "corruption", "condition",
                "names", "codes", "target_slot", "alternate_slot", "candidate_answers", "resample_gold", "generated_text",
                "generation_stop", "exact_clean", "clean_number_content", "first_number_is_clean", "clean_beats_all_candidates")}
            record.update(reference_text=baseline["generated_text"], donor_text=lookup[row["case_id"], "clean"]["generated_text"])
            if failed_baseline:
                result["baseline_failures"].append(record)
            if changed:
                result["changed_focal_outputs"].append(record)
            if row["generation_stop"] == "token_budget":
                result["budget_outputs"].append(record)
    for view in VIEWS:
        for mode in CORRUPTIONS:
            for group in manifest["protocol"]["distance_groups"]:
                paired = [r for r in rows if (r["view"], r["corruption"], r["condition"]) == (view, mode, f"patch:prompt_and_decode:{group}")]
                prompts = [lookup[r["case_id"], f"patch:prompt_only:{group}"] for r in paired]
                if any(a["generated_token_ids"][0] != b["generated_token_ids"][0] or any(
                    abs(a["candidate_scores"][label]["token_logprobs"][0] - b["candidate_scores"][label]["token_logprobs"][0]) > 1e-5
                    for label in a["candidate_scores"]) for a, b in zip(paired, prompts)):
                    raise ValueError("Prompt/combined scopes differ at their identical first intervention")
                result["scope_transitions"].append({"view": view, "corruption": mode, "group": group, "n_identities": len(paired),
                    "same_first_generated_token": len(paired), "new_exact_targets": sum(a["exact_clean"] and not b["exact_clean"] for a, b in zip(paired, prompts)),
                    "lost_exact_targets": sum(b["exact_clean"] and not a["exact_clean"] for a, b in zip(paired, prompts)),
                    "changed_token_sequences": sum(a["generated_token_ids"] != b["generated_token_ids"] for a, b in zip(paired, prompts))})
                for a, b in zip(paired, prompts):
                    if a["exact_clean"] and not b["exact_clean"]:
                        result["scope_repair_examples"].append({"case_id": a["case_id"], "group": group,
                            "candidate_answers": a["candidate_answers"], "prompt_only_text": b["generated_text"], "combined_text": a["generated_text"]})
                if group == "answer_address":
                    for slot in range(4):
                        subset = [r for r in paired if r["target_slot"] == slot]
                        result["target_position_breakdown"].append({"view": view, "corruption": mode, "target_slot_zero_based": slot,
                            "n": len(subset), "count_exact_target": sum(r["exact_clean"] for r in subset),
                            "count_clean_baseline_correct": sum(lookup[r["case_id"], "clean"]["exact_clean"] for r in subset)})
    identities = sorted({e["example_id"] for e in examples})
    for mode in CORRUPTIONS:
        eligible = [identity for identity in identities if all(lookup[f"{identity}/{view}/{mode}", "clean"]["exact_clean"]
                    and lookup[f"{identity}/{view}/{mode}", "corrupt"]["exact_corrupt"] for view in VIEWS)]
        result["common_baseline_success_sensitivity"].append({"corruption": mode, "n": len(eligible), "identities": eligible,
            "count_exact_target": {view: sum(lookup[f"{i}/{view}/{mode}", "patch:prompt_and_decode:answer_address"]["exact_clean"] for i in eligible) for view in VIEWS}})
    for view in VIEWS:
        for scope in SCOPES:
            condition = f"patch:{scope}:answer_address"
            old = [r for r in old_rows if (r["view"], r["corruption"], r["condition"]) == (view, "different_first", condition)]
            old_inputs = [e for e in old_examples if e["view"] == view and e["corruption"] == "different_first"]
            for mode in CORRUPTIONS:
                new = [r for r in rows if (r["view"], r["corruption"], r["condition"]) == (view, mode, condition)]
                new_inputs = [e for e in examples if e["view"] == view and e["corruption"] == mode]
                result["earlier_comparison"].append({"comparison": "descriptive_specificity_vs_multifact", "view": view,
                    "new_corruption": mode, "condition": condition, "old_n": len(old), "new_n": len(new),
                    "old_mean_delta": statistics.fmean(r["delta_clean_logprob"] for r in old),
                    "new_mean_delta": statistics.fmean(r["delta_clean_logprob"] for r in new),
                    "old_count_exact_target": sum(r["exact_clean"] for r in old), "new_count_exact_target": sum(r["exact_clean"] for r in new),
                    "old_recipient_contains_target": sum(e["clean_gold"] in e["corrupt_prompt"] for e in old_inputs),
                    "new_recipient_contains_target": sum(e["clean_gold"] in e["corrupt_prompt"] for e in new_inputs)})
    address = set(map(tuple, manifest["groups"]["answer_address"]))
    result["control_overlaps_with_address"] = {name: sorted(address & set(map(tuple, pairs)))
        for name, pairs in manifest["groups"].items() if name.startswith("random_")}
    if verify_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["protocol"]["model_id"], revision=manifest["model_revision"],
                                                  local_files_only=True, trust_remote_code=False)
        if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_and_chat_template_sha256"]:
            raise ValueError("Cached tokenizer fingerprint differs")
        for stage, ee, refs, rr in (("confirmation", examples, references, rows), ("smoke", smoke_examples, smoke_refs, smoke_rows)):
            if build_examples(tokenizer, manifest["protocol"], stage, prior) != (ee, refs):
                raise ValueError("Exact input/reference regeneration differs")
            if any(tokenizer.decode(r["generated_token_ids"], skip_special_tokens=True) != r["generated_text"] for r in rr):
                raise ValueError("Generated token decoding differs")
        result["audit"]["tokenizer_only_check"] = {"status": "passed", "regenerated_cases": len(examples) + len(smoke_examples),
            "regenerated_reference_prompts": len(references) + len(smoke_refs), "redecoded_generations": len(rows) + len(smoke_rows)}
    return result


def render_figure(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    data = {(r["view"], r["corruption"], r["condition"]): r for r in result["conditions"] if r["family"] == "all"}
    contrasts = {(c["contrast"], c["metric"]): c for c in result["frozen_contrasts"]}
    settings = [(view, mode) for view in VIEWS for mode in CORRUPTIONS]
    labels = ["Far\nValue swap", "Far\nQuery selection", "Near\nValue swap", "Near\nQuery selection"]
    blue, orange, green, grey = "#0072B2", "#D55E00", "#009E73", "#777777"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        ax = axes[0, 0]
        for condition, label, color, offset in (("patch:prompt_only:answer_address", "Three heads: prompt only", blue, -.26),
                                               ("patch:prompt_and_decode:answer_address", "Three heads: prompt + decode", orange, 0),
                                               ("positive_transfer", "Full residual calibration", green, .26)):
            counts = [data[v, m, condition]["count_exact_clean"] for v, m in settings]
            ax.bar([i + offset for i in range(4)], counts, width=.23, color=color, label=label)
            for i, count in enumerate(counts):
                ax.text(i + offset, count + .5, str(count), ha="center", fontsize=9)
        controls = [[r["count_exact_clean"] for r in result["conditions"] if r["family"] == "all" and r["view"] == v
                     and r["corruption"] == m and r["condition"].startswith("patch:prompt_and_decode:random_answer_address_")]
                    for v, m in settings]
        ax.plot(range(4), [statistics.fmean(counts) for counts in controls], "D", color=grey, markersize=5,
                label="Matched-control mean (combined)")
        ax.set(title="A. Sparse patches now switch complete answers", xticks=range(4), xticklabels=labels,
               ylabel="Exact target-code outputs (out of 40)", ylim=(0, 58))
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
        ax = axes[0, 1]
        for scope, label, color, offset in (("prompt_only", "Prompt only", blue, -.05), ("prompt_and_decode", "Prompt + decode", orange, .05)):
            cc = [contrasts[f"patch:{v}:{m}:{scope}:answer_address_minus_controls", "delta_clean_logprob"] for v, m in settings]
            ax.errorbar([i + offset for i in range(4)], [c["mean"] for c in cc],
                yerr=[[c["mean"] - c["ci_low"] for c in cc], [c["ci_high"] - c["mean"] for c in cc]], marker="o", capsize=3, color=color, label=label)
        ax.axhline(0, color=".7", linewidth=.8)
        ax.set(title="B. Score effects exceed matched controls", xticks=range(4), xticklabels=labels,
               ylabel="Target-token log-probability change\n(focal minus controls)", ylim=(0, 3.15))
        ax.legend(frameon=False, fontsize=9)
        ax = axes[1, 0]
        for view, color, offset in (("far_chat", blue, -.12), ("near_chat", orange, .12)):
            c = contrasts[f"task:query_minus_swap:{view}:prompt_and_decode:answer_address:control_adjusted", "delta_exact_clean"]
            ax.errorbar([offset], [100 * c["mean"]], yerr=[[100 * (c["mean"] - c["ci_low"])], [100 * (c["ci_high"] - c["mean"])]],
                        marker="o", capsize=5, color=color, label=view.split("_")[0].capitalize())
        ax.axhline(0, color=".5", linewidth=.9, linestyle="--")
        ax.set(title="C. Query-vs-swap accuracy difference is uncertain", xticks=[-.12, .12], xticklabels=["Far", "Near"],
               ylabel="Query minus swap transfer effect (pp)\n(control-adjusted, combined scope)", xlim=(-.35, .35), ylim=(-15, 35))
        ax = axes[1, 1]
        items = [("clean", "Baseline"), ("zero:prompt_only:non_address_core", "Core\nzero"),
                 ("resample:prompt_only:non_address_core", "Core\nresample"), ("zero:prompt_only:answer_address", "Three heads\nzero"),
                 ("resample:prompt_only:answer_address", "Three heads\nresample")]
        for key, label, color, marker in (("count_exact_clean", "Exact answer", orange, "o"),
                                          ("count_clean_number_content", "Correct-number content", green, "s"),
                                          ("count_clean_beats_all_candidates", "Target ranks first of four", blue, "^")):
            ax.plot(range(5), [data["near_chat", "query_selection", name][key] for name, _ in items],
                    color=color, marker=marker, linewidth=1.1, label=label)
        ax.set(title="D. Scores, content and formatting still differ", xticks=range(5), xticklabels=[label for _, label in items],
               ylabel="Near-view outcomes (out of 40)", ylim=(0, 49))
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("Complete answer switching depends on the task context", fontsize=15)
        fig.supxlabel("40 paired identities; B/C: 95% pointwise identity-bootstrap intervals. A/D: descriptive counts, not independent trials across conditions.\n"
                      "Far value-swap baseline already emits the target once. Calibration copies actual donors, including errors; A plots known target codes.", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"multifact_transfer.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts_publication/multifact_confirmation_8192")
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "artifacts_publication/multifact_smoke_8192")
    parser.add_argument("--earlier-dir", type=Path, default=ROOT / "artifacts_publication/specificity_confirmation_8192")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_publication/multifact_confirmation_8192_diagnostics.json")
    parser.add_argument("--verify-tokenizer", action="store_true")
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run_dir, args.smoke_dir, args.earlier_dir, args.verify_tokenizer)
    write_json(args.out, result)
    if args.figure_dir:
        render_figure(result, args.figure_dir)
    print(json.dumps({"audit": result["audit"], "positive_control": result["positive_control"], "patch_totals": result["patch_totals"]}, indent=2))
    print(f"Wrote multi-fact diagnostics: {args.out}")


if __name__ == "__main__":
    main()
