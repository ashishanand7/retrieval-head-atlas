#!/usr/bin/env python
"""Reproduce specificity audits and separate scores, answer content and stopping."""

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.specificity import ABLATION_VIEWS, DISTANCE_VIEWS, build_examples, prior_ids
from scripts.analyze_publication_specificity import analyze, read_rows
from scripts.compare_publication_pilots import load_checked, sha256
from scripts.diagnose_publication_temporal import load_temporal, summarize_condition as temporal_summary
from scripts.run_publication_confirmation import tokenizer_fingerprint, write_json


def number_diagnostics(row):
    numbers = re.findall(r"(?<!\d)\d{6}(?!\d)", row["generated_text"])
    return {"first_number_is_clean": int(bool(numbers) and numbers[0] == row["clean_gold"]),
            "contains_clean_number": int(row["clean_gold"] in numbers),
            "mentions_decoy": int(row["unrelated_decoy"] in numbers),
            "contains_reference_number": int(row.get("resample_gold") in numbers),
            "first_number_is_reference": int(bool(numbers) and numbers[0] == row.get("resample_gold"))}


def summarize_condition(rows, lookup):
    result = temporal_summary(rows, lookup)
    result.update({f"count_{key}": sum(number_diagnostics(r)[key] for r in rows)
                   for key in number_diagnostics(rows[0])})
    for key in ("verbatim_clean", "generated_leading_whitespace", "exact_resample", "resample_number_content"):
        result[f"count_{key}"] = sum(r[key] for r in rows)
    result["generation_stops"] = dict(Counter(r["generation_stop"] for r in rows))
    result["mean_generated_tokens"] = statistics.fmean(len(r["generated_token_ids"]) for r in rows)
    result["mean_clean_token_logprob"] = [statistics.fmean(r["candidate_scores"]["clean"]["token_logprobs"][i]
                                                        for r in rows) for i in range(6)]
    return result


def load_specificity(directory, stage):
    if not (directory / "rows.jsonl").exists():
        raise FileNotFoundError(f"Restore the retained archive first: gzip -dk '{directory / 'rows.jsonl.gz'}'")
    result = analyze(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["arguments"]["stage"] != stage:
        raise ValueError("Wrong specificity stage")
    for name, value in result.items():
        if json.loads((directory / f"{name}.json").read_text()) != value:
            raise ValueError(f"Saved specificity {name} does not reproduce")
    archive = directory / "rows.jsonl.gz"
    if archive.exists():
        digest = hashlib.sha256()
        with gzip.open(archive, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != manifest["rows_sha256"]:
            raise ValueError("Compressed trace does not match the original run")
    return manifest, read_rows(directory / "rows.jsonl"), read_rows(directory / "examples.jsonl"), read_rows(directory / "references.jsonl"), result


def diagnose(directory, smoke_directory, temporal_directory, earlier_directory, verify_tokenizer=False):
    manifest, rows, examples, references, analysis = load_specificity(directory, "confirmation")
    smoke, smoke_rows, smoke_examples, smoke_refs, smoke_analysis = load_specificity(smoke_directory, "smoke")
    for field in ("protocol", "original_protocol", "source_sha256", "packages", "dtype", "runtime_config", "groups",
                  "model_revision", "tokenizer_and_chat_template_sha256"):
        if manifest[field] != smoke[field]:
            raise ValueError(f"Smoke and full run differ: {field}")
    if manifest["arguments"]["chunk_size"] != smoke["arguments"]["chunk_size"]:
        raise ValueError("Smoke chunk size differs")
    for field in ("rows_sha256", "references_sha256"):
        if manifest["smoke_gate"][field] != smoke[field]:
            raise ValueError("Supplied smoke does not match confirmation gate")
    if manifest["smoke_gate"]["validation"] != smoke_analysis["validation"]:
        raise ValueError("Smoke gate validation differs")
    prior = prior_ids(ROOT, manifest["protocol"]["protocol"])
    schedules = [{(r["variant"], r["seed"]) for r in records}
                 for records in (examples, references, smoke_examples, smoke_refs)]
    if sum(map(len, schedules)) != len(set.union(*schedules)) or set.union(*schedules) & prior:
        raise ValueError("Specificity study/reference schedules overlap each other or prior data")
    temporal, old_rows, _, _ = load_temporal(temporal_directory, "confirmation")
    earlier, earlier_rows, _, _, paired, _ = load_checked(earlier_directory)
    if paired != json.loads((earlier_directory / "paired_analysis.json").read_text()):
        raise ValueError("Earlier confirmation paired analysis differs")
    for meta in (temporal, earlier):
        for field in ("model_revision", "tokenizer_and_chat_template_sha256", "packages", "dtype"):
            if meta[field] != manifest[field]:
                raise ValueError(f"Cross-run model/environment mismatch: {field}")
    if temporal["original_protocol"] != manifest["original_protocol"] or earlier["protocol"] != manifest["original_protocol"]:
        raise ValueError("Cross-run original head configuration differs")
    result = {
        "audit": {"status": "passed", "confirmation": analysis["validation"], "smoke": smoke_analysis["validation"],
                  "all_saved_analyses_reproduce": True, "smoke_gate_matches": True,
                  "compressed_trace_verified": (directory / "rows.jsonl.gz").exists(),
                  "prior_identities_checked": len(prior), "prior_identity_overlap": 0,
                  "study_reference_stage_overlap": 0, "earlier_temporal_and_confirmation_reaudited": True},
        "source_sha256": {f"{label}/{path.name}": sha256(path)
                          for label, folder in (("confirmation", directory), ("smoke", smoke_directory),
                                                ("temporal", temporal_directory), ("earlier", earlier_directory))
                          for path in sorted(folder.iterdir()) if path.suffix in (".json", ".jsonl", ".gz")},
        "analysis_source_sha256": sha256(Path(__file__)),
        "notes": ["Primary inference uses 40 identities. Smoke identities, views, controls and reference donors are not pooled as extra trials.",
                  "Frozen control-adjusted contrasts retain all five draws. First-number and output-transition diagnostics are post-run additions.",
                  "A correct first number can coexist with a later distractor mention; exclusive-number content is a different endpoint.",
                  "Raw exact match is at floor; no accuracy-repair inference follows from a raw-space score change. Budget-limited text is censored.",
                  "Positive transfer reproduces actual donor generation, including formatting and budget stops; this is an output-side calibration, not a sparse mechanism.",
                  "A task-matched resampling donor changes facts, query details and filler. Resampling is neither pure removal nor content-free noise.",
                  "Near/far also changes absolute source position. Short/far changes length and irrelevant context. Pointwise intervals are not multiplicity-adjusted."],
        "runtime": {}, "conditions": [], "smoke_conditions": [], "positive_control": {}, "all_patches": {},
        "baseline_outputs": [], "changed_focal_outputs": [], "earlier_comparison": [],
        "primary_contrasts": analysis["paired_analysis"]["contrasts"], "control_overlaps": {},
    }
    focal = set(manifest["protocol"]["distance_groups"] + manifest["protocol"]["ablation_groups"])
    for label, meta, rr, ee in (("confirmation", manifest, rows, examples), ("smoke", smoke, smoke_rows, smoke_examples)):
        lookup = {(r["case_id"], r["condition"]): r for r in rr}
        buckets = defaultdict(list)
        for row in rr:
            if not row["condition"].startswith("self_patch_"):
                buckets[row["view"], row["corruption"], row["condition"]].append(row)
        summaries = result["conditions" if label == "confirmation" else "smoke_conditions"]
        for (view, mode, condition), selected in sorted(buckets.items()):
            variants = ["all"]
            if label == "confirmation" and (condition in ("clean", "corrupt", "positive_transfer") or condition.split(":")[-1] in focal):
                variants += sorted({r["variant"] for r in selected})
            for variant in variants:
                subset = selected if variant == "all" else [r for r in selected if r["variant"] == variant]
                summaries.append({"view": view, "corruption": mode, "condition": condition, "variant": variant,
                                  **summarize_condition(subset, lookup)})
        positive = [r for r in rr if r["condition"] == "positive_transfer"]
        result["positive_control"][label] = {
            "cases_not_independent_identities": len(positive),
            "generated_token_sequences_equal_donor": sum(r["generated_token_ids"] == lookup[r["case_id"], "clean"]["generated_token_ids"] for r in positive),
            "stopping_conditions_equal_donor": sum(r["generation_stop"] == lookup[r["case_id"], "clean"]["generation_stop"] for r in positive),
            "recorded_current_logit_checks": sum(len(r["generation_steps"]) + sum(len(s["steps"]) for s in r["candidate_scores"].values()) for r in positive),
            "max_logit_error": (analysis if label == "confirmation" else smoke_analysis)["validation"]["positive_max_logit_error"],
            "max_token_score_error": (analysis if label == "confirmation" else smoke_analysis)["validation"]["positive_max_token_score_error"],
            "donor_budget_stops_reproduced": sum(r["generation_stop"] == "token_budget" for r in positive),
        }
        patches = [r for r in rr if r["condition"].startswith("patch:")]
        result["all_patches"][label] = {"condition_rows_not_independent_trials": len(patches),
            **{f"count_{key}": sum(r[key] for r in patches) for key in ("exact_clean", "clean_number_content", "exact_decoy")},
            "count_contains_clean_number": sum(number_diagnostics(r)["contains_clean_number"] for r in patches),
            "maximum_answer_logprob_margin": max(r["answer_logprob_margin"] for r in patches),
            "control_token_sequences_changed": sum(r["generated_token_ids"] != lookup[r["case_id"], "corrupt"]["generated_token_ids"]
                                                    for r in patches if ":random_" in r["condition"])}
        result["runtime"][label] = {"elapsed_minutes": meta["elapsed_seconds_total"] / 60,
            "peak_allocated_cuda_gib": meta["peak_cuda_memory_bytes"] / 1024 ** 3,
            "generation_stops": dict(Counter(r["generation_stop"] for r in rr)),
            "views": {view: {"prompt_tokens": [min(e["prompt_tokens"] for e in ee if e["view"] == view), max(e["prompt_tokens"] for e in ee if e["view"] == view)],
                             "source_to_question_tokens": [min(e["source_to_question_tokens"] for e in ee if e["view"] == view), max(e["source_to_question_tokens"] for e in ee if e["view"] == view)]}
                      for view in ABLATION_VIEWS}}
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    for row in rows:
        baseline = lookup[row["case_id"], row["reference"]]
        is_baseline = row["condition"] in ("clean", "corrupt")
        if is_baseline or (row["condition"].split(":")[-1] in focal and row["generated_token_ids"] != baseline["generated_token_ids"]):
            record = {key: row[key] for key in ("case_id", "example_id", "variant", "view", "corruption", "condition",
                      "clean_gold", "corrupt_gold", "unrelated_decoy", "resample_gold", "generated_text", "generation_stop",
                      "exact_clean", "clean_number_content", "exact_decoy", "exact_resample", "resample_number_content")}
            record.update(reference_text=baseline["generated_text"], reference_stop=baseline["generation_stop"], **number_diagnostics(row))
            result["baseline_outputs" if is_baseline else "changed_focal_outputs"].append(record)
    address = set(map(tuple, manifest["groups"]["answer_address"]))
    result["control_overlaps"] = {name: sorted(address & set(map(tuple, pairs))) for name, pairs in manifest["groups"].items() if name.startswith("random_")}
    for mode in ("different_first", "same_first"):
        for group in manifest["protocol"]["distance_groups"]:
            for scope in ("prompt_only", "prompt_and_decode"):
                condition = f"patch:{scope}:{group}"
                new = [r for r in rows if r["view"] == "far_chat" and r["corruption"] == mode and r["condition"] == condition]
                old = [r for r in old_rows if r["corruption"] == mode and r["condition"] == condition]
                result["earlier_comparison"].append({"comparison": "temporal_vs_specificity_far_chat", "corruption": mode, "condition": condition,
                    "old_n": len(old), "new_n": len(new), "old_mean_delta": statistics.fmean(r["delta_clean_logprob"] for r in old),
                    "new_mean_delta": statistics.fmean(r["delta_clean_logprob"] for r in new),
                    "old_exact_donor": sum(r["exact_clean"] for r in old), "new_exact_donor": sum(r["exact_clean"] for r in new)})
    for group in manifest["protocol"]["ablation_groups"]:
        old = [r for r in earlier_rows if r["marker_mode"] == "unmarked" and r["condition"] == f"ablate:{group}"]
        new = [r for r in rows if r["view"] == "far_chat" and r["corruption"] == "different_first" and r["condition"] == f"zero:prompt_only:{group}"]
        result["earlier_comparison"].append({"comparison": "confirmation_vs_specificity_query_zero", "group": group,
            "old_n": len(old), "new_n": len(new), "old_mean_delta": statistics.fmean(r["delta_clean_logprob"] for r in old),
            "new_mean_delta": statistics.fmean(r["delta_clean_logprob"] for r in new),
            "old_exact_clean": sum(r["exact_clean"] for r in old), "new_exact_clean": sum(r["exact_clean"] for r in new)})
    if verify_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["protocol"]["model_id"], revision=manifest["model_revision"],
                                                  local_files_only=True, trust_remote_code=False)
        if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_and_chat_template_sha256"]:
            raise ValueError("Cached tokenizer fingerprint differs")
        for stage, ee, refs, rr in (("confirmation", examples, references, rows), ("smoke", smoke_examples, smoke_refs, smoke_rows)):
            if build_examples(tokenizer, manifest["protocol"], stage, prior) != (ee, refs):
                raise ValueError("Exact example/reference regeneration differs")
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
    contrasts = {(c["contrast"], c["metric"]): c for c in result["primary_contrasts"]}
    conditions = {(r["view"], r["corruption"], r["condition"]): r for r in result["conditions"] if r["variant"] == "all"}
    blue, red, green = "#0072B2", "#D55E00", "#009E73"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), layout="constrained")
        ax = axes[0, 0]
        for scope, label, color, offset in (("prompt_only", "Prompt only", blue, -.04), ("prompt_and_decode", "Prompt + decode", red, .04)):
            cc = [contrasts[f"distance:{view}:different_first:{scope}:answer_address_minus_controls", "delta_clean_logprob"] for view in DISTANCE_VIEWS]
            ax.errorbar([i + offset for i in range(3)], [c["mean"] for c in cc],
                        yerr=[[c["mean"] - c["ci_low"] for c in cc], [c["ci_high"] - c["mean"] for c in cc]],
                        marker="o", capsize=3, color=color, label=label)
        ax.set(title="A. Three-head score effects persist locally", xticks=range(3), xticklabels=["Far, 8K", "Near, 8K", "Short"],
               ylabel="Donor-token log-probability change\n(focal minus matched controls)", ylim=(0, 3))
        ax.legend(frameon=False, fontsize=9)
        ax = axes[0, 1]
        for condition, label, color, offset in (("patch:prompt_and_decode:answer_address", "Three-head patch", red, -.18),
                                                 ("positive_transfer", "Full residual calibration", green, .18)):
            values = [conditions[view, "different_first", condition]["count_exact_clean"] for view in DISTANCE_VIEWS]
            ax.bar([i + offset for i in range(3)], values, width=.34, label=label, color=color)
            for i, value in enumerate(values):
                ax.text(i + offset, value + .7, str(value), ha="center", fontsize=9)
        ax.set(title="B. Complete donor answers: patch vs calibration", xticks=range(3), xticklabels=["Far, 8K", "Near, 8K", "Short"],
               ylabel="Exact donor answers (out of 40)", ylim=(0, 55))
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax = axes[1, 0]
        views = ("far_raw", "far_raw_space", "far_chat")
        for action, label, color, offset in (("zero", "Zero ablation", red, -.04), ("resample", "Task-matched resampling", blue, .04)):
            cc = [contrasts[f"ablation:{view}:{action}:non_address_core_minus_controls", "delta_clean_logprob"] for view in views]
            ax.errorbar([i + offset for i in range(3)], [c["mean"] for c in cc],
                        yerr=[[c["mean"] - c["ci_low"] for c in cc], [c["ci_high"] - c["mean"] for c in cc]],
                        marker="o", capsize=3, color=color, label=label)
        ax.axhline(0, color=".7", linewidth=.8)
        ax.set(title="C. Core-group penalty depends on answer boundary", xticks=range(3), xticklabels=["Raw", "Raw + space", "Chat"],
               ylabel="Clean-token log-probability change\n(focal minus matched controls)", ylim=(-1.3, .4))
        ax.legend(frameon=False, fontsize=9)
        ax = axes[1, 1]
        for group, action, label, color, marker, offset in (("non_address_core", "zero", "Core: zero", red, "o", -.04),
                                                           ("non_address_core", "resample", "Core: resample", green, "s", .04),
                                                           ("answer_address", "resample", "Three heads: resample", blue, "^", 0)):
            values = [conditions[view, "different_first", f"{action}:prompt_only:{group}"]["count_clean_number_content"] for view in DISTANCE_VIEWS]
            ax.plot([i + offset for i in range(3)], values, marker=marker, color=color, label=label, markersize=5)
        ax.set(title="D. Resampling is not interchangeable with zeroing", xticks=range(3), xticklabels=["Far, 8K", "Near, 8K", "Short"],
               ylabel="Correct-number content (out of 40)", ylim=(0, 43))
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        fig.suptitle("Specificity controls narrow the interpretation of head importance", fontsize=15)
        fig.supxlabel("Different-first-digit cases; 40 paired identities and five matched control draws. A/C: 95% pointwise bootstrap intervals.\n"
                      "B/D: observed counts, not independent trials across views. The full-residual control is not a sparse circuit.", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"specificity_controls.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts_publication/specificity_confirmation_8192")
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "artifacts_publication/specificity_smoke_8192")
    parser.add_argument("--temporal-dir", type=Path, default=ROOT / "artifacts_publication/temporal_confirmation_8192_chat")
    parser.add_argument("--earlier-dir", type=Path, default=ROOT / "artifacts_publication/confirmation_8192_chat")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_publication/specificity_confirmation_8192_diagnostics.json")
    parser.add_argument("--verify-tokenizer", action="store_true", help="Regenerate inputs and decode outputs using only the cached pinned tokenizer")
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run_dir, args.smoke_dir, args.temporal_dir, args.earlier_dir, args.verify_tokenizer)
    write_json(args.out, result)
    if args.figure_dir:
        render_figure(result, args.figure_dir)
    print(json.dumps({"audit": result["audit"]["status"], "positive_control": result["positive_control"], "all_patches": result["all_patches"]}, indent=2))
    print(f"Wrote specificity diagnostics: {args.out}")


if __name__ == "__main__":
    main()
