#!/usr/bin/env python
"""Validate temporal traces and compute paired, identity-level scope contrasts."""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.confirmation import VARIANTS
from rha.temporal import CORRUPTIONS, SCOPES, condition_names, output_metrics, select_groups
from rha.temporal_scoring import patch_active
from scripts.analyze_publication_confirmation import paired_interval
from scripts.run_publication_confirmation import write_json
from scripts.run_publication_temporal import assert_self_patch, verify_committed_sources

METRICS = ("clean_mean_logprob", "answer_logprob_margin", "clean_decoy_logprob_margin",
           "delta_clean_logprob", "delta_answer_margin", "delta_clean_decoy_margin", "delta_exact_clean",
           "exact_clean", "exact_corrupt", "exact_decoy", "six_digits_only", "clean_number_content",
           "first_token_is_clean", "first_differing_token_is_clean", "aligned_suffix_exact_clean",
           "aligned_suffix_fraction_clean", "aligned_suffix_exact_corrupt")


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Inconsistent temporal {label}")


def validate_rows(manifest, examples, rows):
    if not finite(rows):
        raise ValueError("Non-finite temporal readout")
    by_case = {example["case_id"]: example for example in examples}
    lookup = {(row["case_id"], row["condition"]): row for row in rows}
    expected = {(case, condition) for case in by_case for condition in condition_names(manifest["groups"])}
    identities = {example["example_id"] for example in examples}
    if (len(by_case) != len(examples) or len(lookup) != len(rows) or set(lookup) != expected
            or len(rows) != manifest["rows_written"] or len(rows) != manifest["expected_rows"]
            or len(identities) != manifest["n_independent_example_ids"] or len(examples) != manifest["n_example_pairs"]):
        raise ValueError("Incomplete or duplicate temporal condition grid")
    if {(ex["example_id"], ex["corruption"]) for ex in examples} != {(identity, mode) for identity in identities for mode in CORRUPTIONS}:
        raise ValueError("Missing paired corruption stratum")
    union = sorted({tuple(pair) for heads in manifest["groups"].values() for pair in heads})
    self_error = 0.0
    for example in examples:
        for label in ("clean", "corrupt"):
            if hashlib.sha256(example[f"{label}_prompt"].encode()).hexdigest() != example[f"{label}_prompt_sha256"]:
                raise ValueError("Temporal prompt hash mismatch")
        clean, corrupt = example["clean_prompt"], example["corrupt_prompt"]
        if clean.replace(example["clean_gold"], "<ANSWER>") != corrupt.replace(example["corrupt_gold"], "<ANSWER>"):
            raise ValueError("Temporal corruption changed more than the answer")
        ids = example["candidate_token_ids"]
        differing = next(i for i, pair in enumerate(zip(ids["clean"], ids["corrupt"])) if pair[0] != pair[1])
        if differing != example["shared_prefix_tokens"] or differing != CORRUPTIONS.index(example["corruption"]):
            raise ValueError("Temporal common-prefix stratum mismatch")
    for identity in identities:
        pair = [example for example in examples if example["example_id"] == identity]
        for field in ("clean_prompt", "clean_gold", "unrelated_decoy", "filler_words", "variant", "seed"):
            if pair[0][field] != pair[1][field]:
                raise ValueError("Corruption strata do not share the clean example")
    for row in rows:
        example = by_case[row["case_id"]]
        for key, value in example.items():
            if key not in ("clean_prompt", "corrupt_prompt") and row[key] != value:
                raise ValueError(f"Temporal row/example mismatch: {key}")
        condition = row["condition"]
        if condition in ("clean", "corrupt"):
            scope, pairs, reference = "none", [], condition
        elif condition.startswith("self_patch_"):
            name, scope = condition.split(":")
            pairs, reference = union, name.removeprefix("self_patch_")
        else:
            _, scope, name = condition.split(":")
            pairs, reference = manifest["groups"][name], "corrupt"
        if row["reference"] != reference:
            raise ValueError("Incorrect temporal reference baseline")
        has_donor = scope != "none"
        if set(row["candidate_scores"]) != {"clean", "corrupt", "decoy"}:
            raise ValueError("Missing temporal candidate scores")
        for label, score in row["candidate_scores"].items():
            if label not in ("clean", "corrupt", "decoy") or score["token_ids"] != example["candidate_token_ids"][label]:
                raise ValueError("Teacher-forced candidate tokens changed")
            if len(score["token_logprobs"]) != len(score["token_ids"]) or score["stop"] != "teacher_forced":
                raise ValueError("Invalid teacher-forced score length or termination")
            close(score["sum_logprob"], sum(score["token_logprobs"]), "token sum")
            close(score["mean_logprob"], score["sum_logprob"] / len(score["token_ids"]), "token mean")
            close(row[f"{label}_mean_logprob"], score["mean_logprob"], "candidate mean")
        scores = row["candidate_scores"]
        close(row["answer_logprob_margin"], scores["clean"]["sum_logprob"] - scores["corrupt"]["sum_logprob"], "answer margin")
        close(row["clean_decoy_logprob_margin"], scores["clean"]["sum_logprob"] - scores["decoy"]["sum_logprob"], "distractor margin")
        close(row["first_clean_logprob"], scores["clean"]["token_logprobs"][0], "first-token score")
        close(row["first_answer_logprob_margin"], scores["clean"]["token_logprobs"][0] - scores["corrupt"]["token_logprobs"][0], "first-token margin")
        tokens = row["generated_token_ids"]
        if not tokens or len(tokens) != len(row["generated_token_logprobs"]) or len(tokens) > manifest["protocol"]["max_new_tokens"]:
            raise ValueError("Invalid generation length")
        eos = set(manifest["eos_token_ids"])
        if any(token in eos for token in tokens[:-1]):
            raise ValueError("Generation continued after EOS")
        if row["generation_stop"] == "eos":
            if tokens[-1] not in eos:
                raise ValueError("EOS termination without an EOS token")
        elif row["generation_stop"] != "token_budget" or tokens[-1] in eos or len(tokens) != manifest["protocol"]["max_new_tokens"]:
            raise ValueError("Incorrect generation termination")
        branches = [(score["token_ids"], score["steps"]) for score in scores.values()]
        branches.append((tokens, row["generation_steps"]))
        for branch_tokens, steps in branches:
            if len(steps) != len(branch_tokens):
                raise ValueError("Missing temporal step trace")
            for index, event in enumerate(steps):
                input_id = example["final_prompt_token_id"] if index == 0 else branch_tokens[index - 1]
                position = example["prompt_tokens"] - 1 + index
                expected_event = {
                    "step": index, "recipient_input_token_id": input_id,
                    "donor_input_token_id": input_id if has_donor else None,
                    "recipient_cache_length_before": position, "donor_cache_length_before": position if has_donor else None,
                    "patch_applied": patch_active(scope, index),
                    "verified_hook_calls": {str(layer): int(patch_active(scope, index)) for layer, _ in pairs},
                }
                if event != expected_event:
                    raise ValueError("Temporal prefix, position, scope, or hook trace mismatch")
        answers = {"clean": row["clean_gold"], "corrupt": row["corrupt_gold"], "decoy": row["unrelated_decoy"]}
        for key, expected_value in output_metrics(row["generated_text"], tokens, answers, example["candidate_token_ids"]).items():
            if row[key] != expected_value:
                raise ValueError(f"Temporal generation metric mismatch: {key}")
        baseline = lookup[row["case_id"], reference]
        if scope == "decode_only":
            if tokens[0] != baseline["generated_token_ids"][0]:
                raise ValueError("Decoding-only patch changed the first generated token")
            for label in scores:
                if abs(scores[label]["token_logprobs"][0] - baseline["candidate_scores"][label]["token_logprobs"][0]) > 1e-5:
                    raise ValueError("Decoding-only patch changed an unpatched first-token score")
        for metric, delta in (("clean_mean_logprob", "delta_clean_logprob"), ("answer_logprob_margin", "delta_answer_margin"),
                              ("clean_decoy_logprob_margin", "delta_clean_decoy_margin"), ("exact_clean", "delta_exact_clean")):
            close(row[delta], row[metric] - baseline[metric], "paired delta")
        if condition.startswith("self_patch_"):
            assert_self_patch(row, baseline)
            self_error = max(self_error, *(abs(a - b) for label in scores for a, b in
                                          zip(scores[label]["token_logprobs"], baseline["candidate_scores"][label]["token_logprobs"])),
                             *(abs(a - b) for a, b in zip(row["generated_token_logprobs"], baseline["generated_token_logprobs"])))
    baseline_counts = {f"{mode}:{label}": sum(row[f"exact_{label}"] for row in rows
                                             if row["corruption"] == mode and row["condition"] == label)
                       for mode in CORRUPTIONS for label in ("clean", "corrupt")}
    return {"status": "passed", "rows": len(rows), "independent_identities": len(identities),
            "prompt_pair_cases": len(examples), "self_patch_max_token_score_error": self_error,
            "prefix_position_scope_hook_trace_checks": "passed", "baseline_exact_counts": baseline_counts}


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for variant in ("all", row["variant"]):
            buckets[row["corruption"], row["condition"], variant].append(row)
    return {"by_condition": [{"corruption": mode, "condition": condition, "variant": variant,
                               "n_identities": len(values), **{f"mean_{metric}": statistics.fmean(row[metric] for row in values) for metric in METRICS},
                               "token_budget_stops": sum(row["generation_stop"] == "token_budget" for row in values)}
                              for (mode, condition, variant), values in sorted(buckets.items())]}


def paired_analysis(manifest, rows, draws):
    lookup = {(row["example_id"], row["corruption"], row["condition"]): row for row in rows}
    identities = sorted({row["example_id"] for row in rows})
    variants = [lookup[identity, CORRUPTIONS[0], "clean"]["variant"] for identity in identities]
    contrasts = []

    def contrast(label, metric, left_mode, left, right_mode, right):
        values = [statistics.fmean(lookup[identity, left_mode, condition][metric] for condition in left)
                  - statistics.fmean(lookup[identity, right_mode, condition][metric] for condition in right)
                  for identity in identities]
        contrasts.append({"contrast": label, "metric": metric, **paired_interval(values, variants, draws)})

    for group in manifest["protocol"]["focal_groups"]:
        controls = sorted(name for name in manifest["groups"] if name.startswith(f"random_{group}_"))
        if not controls:
            raise ValueError("Missing temporal matched controls")
        for metric in ("delta_clean_logprob", "delta_exact_clean", "delta_clean_decoy_margin",
                       "first_differing_token_is_clean", "aligned_suffix_fraction_clean"):
            for mode in CORRUPTIONS:
                for scope in SCOPES:
                    focal = [f"patch:{scope}:{group}"]
                    contrast(f"{mode}:{scope}:{group}_minus_controls", metric, mode, focal, mode,
                             [f"patch:{scope}:{name}" for name in controls])
                    contrast(f"{mode}:{scope}:{group}_minus_corrupt", metric, mode, focal, mode, ["corrupt"])
                for scope in ("decode_only", "prompt_and_decode"):
                    contrast(f"{mode}:{group}:{scope}_minus_prompt_only", metric, mode, [f"patch:{scope}:{group}"],
                             mode, [f"patch:prompt_only:{group}"])
            for scope in SCOPES:
                contrast(f"{scope}:{group}:same_first_minus_different_first", metric,
                         "same_first", [f"patch:{scope}:{group}"], "different_first", [f"patch:{scope}:{group}"])
    return {"bootstrap_draws": draws, "method": "Paired percentile bootstrap, resampling identities within variant. 95% pointwise intervals; fixed controls averaged within identity. Corruption types and scopes are paired repetitions, not extra identities.",
            "warning": "Diagnostic sample size, no multiplicity adjustment. Zero-width bootstrap intervals for zero events do not establish a zero population rate. Donor answers in corrupt contexts are transfer, not recipient accuracy.",
            "contrasts": contrasts}


def analyze(directory, draws=4000):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only analyze a completed temporal run")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"]):
        raise ValueError("Expected a full temporal experiment Git revision")
    verify_committed_sources(manifest["source_sha256"], manifest["git_commit"])
    for name, field in (("rows.jsonl", "rows_sha256"), ("examples.jsonl", "examples_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f"Temporal input hash mismatch: {name}")
    expected_groups = select_groups(manifest["protocol"], manifest["original_protocol"])
    if {name: list(map(tuple, heads)) for name, heads in manifest["groups"].items()} != expected_groups:
        raise ValueError("Temporal groups differ from the fixed original draws")
    if manifest["model_revision"] != manifest["protocol"]["revision"] or manifest["tokenizer_and_chat_template_sha256"] != manifest["protocol"]["tokenizer_and_chat_template_sha256"]:
        raise ValueError("Temporal model/tokenizer revision mismatch")
    examples, rows = read_rows(directory / "examples.jsonl"), read_rows(directory / "rows.jsonl")
    settings = manifest["protocol"]["stages"][manifest["arguments"]["stage"]]
    expected_ids = {f"{variant}:{settings['seed_base'] + index * 100000 + offset}"
                    for index, variant in enumerate(VARIANTS) for offset in range(settings["n_per_variant"])}
    if ({example["example_id"] for example in examples} != expected_ids
            or manifest["resolved_seed_base"] != settings["seed_base"] or manifest["n_per_variant"] != settings["n_per_variant"]):
        raise ValueError("Temporal identities differ from the reserved stage schedule")
    for example in examples:
        if example["example_id"] != f"{example['variant']}:{example['seed']}":
            raise ValueError("Temporal identity and variant metadata disagree")
        if (example["marker_mode"], example["prompt_format"]) != ("unmarked", "chat"):
            raise ValueError("Temporal prompt interface is not unmarked chat")
        if example["target_tokens"] != manifest["protocol"]["target_tokens"] or example["needle_frac"] != manifest["protocol"]["needle_frac"]:
            raise ValueError("Temporal context settings differ from protocol")
        if example["prompt_tokens"] > manifest["protocol"]["max_len"]:
            raise ValueError("Overlength saved temporal prompt")
    result = {"validation": validate_rows(manifest, examples, rows), "summary": summarize(rows),
              "paired_analysis": paired_analysis(manifest, rows, draws)}
    if (directory / "summary.json").exists() and json.loads((directory / "summary.json").read_text()) != result["summary"]:
        raise ValueError("Saved temporal summary does not reproduce")
    return result


def write_analysis(directory, result):
    for name, value in result.items():
        write_json(directory / f"{name}.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.bootstrap_draws)
    write_analysis(args.run_dir, result)
    print(json.dumps(result["validation"], indent=2))
    print(f"Wrote summary and {len(result['paired_analysis']['contrasts'])} paired contrasts: {args.run_dir}")


if __name__ == "__main__":
    main()
