#!/usr/bin/env python
"""Audit specificity artifacts and compute within-identity panel interactions."""

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.specificity import (ABLATION_VIEWS, BOUNDARY_VIEWS, DISTANCE_VIEWS, SCOPES, VIEW_CORRUPTIONS, boundary_metrics,
                             condition_specs, digest, make_body, prior_ids, reserved_ids, select_groups)
from rha.temporal import CORRUPTIONS, output_metrics
from rha.temporal_scoring import patch_active
from rha.prompts import seeded_decoy, seeded_secret
from scripts.analyze_publication_confirmation import paired_interval
from scripts.analyze_publication_temporal import METRICS as TEMPORAL_METRICS, close, finite, read_rows
from scripts.run_publication_confirmation import write_json
from scripts.run_publication_specificity import DELTA_FIELDS, DELTA_NAMES
from scripts.run_publication_temporal import assert_self_patch, verify_committed_sources

METRICS = tuple(dict.fromkeys((*TEMPORAL_METRICS, *DELTA_NAMES, "verbatim_clean", "generated_leading_whitespace",
                              "exact_resample", "resample_number_content")))


def validate_design(manifest, examples, references):
    protocol = manifest["protocol"]
    ids = {ex["example_id"] for ex in examples}
    by_case = {ex["case_id"]: ex for ex in examples}
    refs = {ref["reference_id"]: ref for ref in references}
    if (len(by_case) != len(examples) or len(refs) != len(references)
            or len(ids) != manifest["n_independent_example_ids"] or len(examples) != manifest["n_example_pairs"]
            or len(references) != manifest["n_reference_prompts"]
            or len({(r["variant"], r["seed"]) for r in references}) != manifest["n_reference_identities"]
            or {(e["example_id"], e["view"], e["corruption"]) for e in examples}
            != {(identity, view, mode) for identity in ids for view, mode in VIEW_CORRUPTIONS}):
        raise ValueError("Incomplete or duplicate specificity example/reference grid")
    wrappers = set()

    def check_render(prompt, body, view):
        if view.endswith("_chat"):
            if prompt.count(body) != 1:
                raise ValueError("Chat prompt does not contain the expected task body")
            prefix, suffix = prompt.split(body)
            if not prefix or not suffix:
                raise ValueError("Missing chat wrapper")
            wrappers.add((prefix, suffix))
        elif prompt != body + (" " if view == "far_raw_space" else ""):
            raise ValueError("Raw boundary differs beyond the specified space")

    for ex in examples:
        identity, view, mode = ex["example_id"], ex["view"], ex["corruption"]
        if identity != f"{ex['variant']}:{ex['seed']}" or ex["case_id"] != f"{identity}/{view}/{mode}":
            raise ValueError("Incorrect identity metadata")
        body = make_body(ex["variant"], ex["seed"], ex["filler_words"], view.split("_")[0])
        check_render(ex["clean_prompt"], body, view)
        gold, other, decoy = ex["clean_gold"], ex["corrupt_gold"], ex["unrelated_decoy"]
        if (gold != seeded_secret(ex["seed"]) or decoy != seeded_decoy(ex["seed"])
                or len({gold, other, decoy}) != 3 or any(re.fullmatch(r"[1-9][0-9]{5}", value) is None for value in (gold, other, decoy))
                or ex["clean_prompt"].count(gold) != 1 or ex["corrupt_prompt"].count(other) != 1
                or ex["clean_prompt"].replace(gold, other) != ex["corrupt_prompt"]):
            raise ValueError("Invalid answer-only corruption")
        for label in ("clean", "corrupt"):
            if digest(ex[f"{label}_prompt"]) != ex[f"{label}_prompt_sha256"]:
                raise ValueError("Prompt hash mismatch")
        if (ex["gold_char_start"] != ex["clean_prompt"].index(gold)
                or ex["corrupt_gold_char_start"] != ex["corrupt_prompt"].index(other)):
            raise ValueError("Wrong character span")
        ids_by_label = ex["candidate_token_ids"]
        if set(ids_by_label) != {"clean", "corrupt", "decoy"} or any(len(value) != 6 for value in ids_by_label.values()):
            raise ValueError("Invalid six-digit candidate tokens")
        differences = [i for i, (a, b) in enumerate(zip(ids_by_label["clean"], ids_by_label["corrupt"])) if a != b]
        shared = CORRUPTIONS.index(mode)
        if (not differences or differences[0] != shared or ex["shared_prefix_tokens"] != shared
                or gold[:shared] != other[:shared] or gold[shared] == other[shared]
                or ex["changed_token_first"] != ex["source_token_start"] + differences[0]
                or ex["changed_token_last"] != ex["source_token_start"] + differences[-1]
                or ex["first_answer_tokens_differ"] != (shared == 0)):
            raise ValueError("Corruption/token-span stratum mismatch")
        if (ex["answer_tokens"] != 6 or ex["source_token_end"] != ex["source_token_start"] + 6
                or not 0 <= ex["source_token_start"] < ex["source_token_end"] < ex["question_token_start"] < ex["prompt_tokens"] <= protocol["max_len"]
                or ex["source_to_question_tokens"] != ex["question_token_start"] - ex["source_token_end"]
                or ex["source_to_answer_tokens"] != ex["prompt_tokens"] - ex["source_token_end"]):
            raise ValueError("Invalid source/query distance metadata")
        expected_target = None if view == "short_chat" else protocol["target_tokens"]
        if ex["marker_mode"] != "unmarked" or ex["target_tokens"] != expected_target or (view == "short_chat" and ex["filler_words"] != 0):
            raise ValueError("Incorrect task/length metadata")
        has_reference = view in ABLATION_VIEWS and mode == "different_first"
        if has_reference:
            ref = refs.get(ex["reference_id"])
            if (ref is None or ref["paired_example_id"] != identity or ref["view"] != view or ref["variant"] != ex["variant"]
                    or ref["seed"] != ex["seed"] + protocol["reference_seed_offset"] or ref["gold"] != ex["resample_gold"]
                    or ref["gold"] in (gold, other, decoy)):
                raise ValueError("Incorrect task-matched resampling reference")
        elif ex["reference_id"] is not None or ex["resample_gold"] is not None:
            raise ValueError("Unexpected resampling reference outside ablation cases")
    if {e["reference_id"] for e in examples if e["reference_id"]} != set(refs):
        raise ValueError("Missing or unused reference prompts")
    for ref in references:
        if (ref["reference_id"] != f"{ref['variant']}:{ref['seed']}/{ref['view']}"
                or ref["gold"] != seeded_secret(ref["seed"]) or ref["decoy"] != seeded_decoy(ref["seed"])
                or digest(ref["prompt"]) != ref["prompt_sha256"] or not 2 <= ref["prompt_tokens"] <= protocol["max_len"]):
            raise ValueError("Invalid reference provenance")
        check_render(ref["prompt"], make_body(ref["variant"], ref["seed"], ref["filler_words"], ref["view"].split("_")[0]), ref["view"])
    if len(wrappers) != 1:
        raise ValueError("Chat wrappers differ across views or references")
    for identity in ids:
        by_view = {(e["view"], e["corruption"]): e for e in examples if e["example_id"] == identity}
        for mode in CORRUPTIONS:
            pair = [e for (view, corruption), e in by_view.items() if corruption == mode]
            for key in ("clean_gold", "corrupt_gold", "unrelated_decoy", "candidate_token_ids"):
                if any(e[key] != pair[0][key] for e in pair):
                    raise ValueError("Views do not share the same answer pair")
            far, near = by_view["far_chat", mode], by_view["near_chat", mode]
            if (far["prompt_tokens"] != near["prompt_tokens"] or far["filler_words"] != near["filler_words"]
                    or far["source_to_question_tokens"] <= near["source_to_question_tokens"]):
                raise ValueError("Invalid matched-length distance comparison")
        for view in DISTANCE_VIEWS:
            if by_view[view, "different_first"]["clean_prompt"] != by_view[view, "same_first"]["clean_prompt"]:
                raise ValueError("Corruptions do not share the same clean prompt")
        far = by_view["far_chat", "different_first"]
        for view in BOUNDARY_VIEWS:
            if by_view[view, "different_first"]["filler_words"] != far["filler_words"]:
                raise ValueError("Boundary condition changed the filler")


def validate_rows(manifest, examples, references, rows):
    if not finite(rows):
        raise ValueError("Non-finite specificity readout")
    validate_design(manifest, examples, references)
    by_case = {e["case_id"]: e for e in examples}
    refs = {r["reference_id"]: r for r in references}
    specs = {e["case_id"]: condition_specs(e, manifest["groups"], manifest["protocol"]) for e in examples}
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    expected = {(case, name) for case, conditions in specs.items() for name in conditions}
    if (len(lookup) != len(rows) or set(lookup) != expected
            or len(rows) != manifest["rows_written"] or len(rows) != manifest["expected_rows"]):
        raise ValueError("Incomplete or duplicate specificity condition grid")
    max_self_error, max_positive_error, max_logit_error = 0.0, 0.0, 0.0
    for row in rows:
        ex, spec = by_case[row["case_id"]], specs[row["case_id"]][row["condition"]]
        if any(row[key] != value for key, value in ex.items() if key not in ("clean_prompt", "corrupt_prompt")):
            raise ValueError("Row/example metadata mismatch")
        if row["reference"] != spec["recipient"]:
            raise ValueError("Incorrect reference baseline")
        scores = row["candidate_scores"]
        if set(scores) != {"clean", "corrupt", "decoy"}:
            raise ValueError("Missing candidate scores")
        for label, score in scores.items():
            if score["token_ids"] != ex["candidate_token_ids"][label] or len(score["token_logprobs"]) != 6 or score["stop"] != "teacher_forced":
                raise ValueError("Invalid candidate scoring branch")
            close(score["sum_logprob"], sum(score["token_logprobs"]), "token sum")
            close(score["mean_logprob"], score["sum_logprob"] / 6, "token mean")
            close(row[f"{label}_mean_logprob"], score["mean_logprob"], "candidate mean")
        close(row["answer_logprob_margin"], scores["clean"]["sum_logprob"] - scores["corrupt"]["sum_logprob"], "answer margin")
        close(row["clean_decoy_logprob_margin"], scores["clean"]["sum_logprob"] - scores["decoy"]["sum_logprob"], "decoy margin")
        close(row["first_clean_logprob"], scores["clean"]["token_logprobs"][0], "first score")
        close(row["first_answer_logprob_margin"], scores["clean"]["token_logprobs"][0] - scores["corrupt"]["token_logprobs"][0], "first margin")
        tokens, eos = row["generated_token_ids"], set(manifest["eos_token_ids"])
        if (not tokens or len(tokens) != len(row["generated_token_logprobs"]) or len(tokens) > manifest["protocol"]["max_new_tokens"]
                or any(token in eos for token in tokens[:-1])):
            raise ValueError("Invalid generated token sequence")
        if row["generation_stop"] == "eos":
            if tokens[-1] not in eos:
                raise ValueError("Missing terminal EOS")
        elif row["generation_stop"] != "token_budget" or tokens[-1] in eos or len(tokens) != manifest["protocol"]["max_new_tokens"]:
            raise ValueError("Incorrect generation stop")
        for branch_tokens, steps in [(score["token_ids"], score["steps"]) for score in scores.values()] + [(tokens, row["generation_steps"])]:
            if len(steps) != len(branch_tokens):
                raise ValueError("Missing step trace")
            for index, event in enumerate(steps):
                token = ex["final_prompt_token_id"] if index == 0 else branch_tokens[index - 1]
                position, active = ex["prompt_tokens"] - 1 + index, patch_active(spec["scope"], index)
                dynamic_donor = spec["donor"] in ("clean", "corrupt")
                expected_event = {"step": index, "recipient_input_token_id": token,
                    "donor_input_token_id": token if dynamic_donor else None,
                    "recipient_cache_length_before": position, "donor_cache_length_before": position if dynamic_donor else None,
                    "patch_applied": active, "verified_hook_calls": {str(layer): int(active) for layer, _ in spec["pairs"]}}
                if spec["action"] in ("zero", "resample", "residual"):
                    error = event.get("donor_logit_max_abs_error")
                    residual = spec["action"] == "residual"
                    if residual:
                        if not isinstance(error, (int, float)) or not 0 <= error <= 1e-5:
                            raise ValueError("Positive transfer logit check failed")
                        max_logit_error = max(max_logit_error, error)
                    elif error is not None:
                        raise ValueError("Unexpected donor logit comparison")
                    source = refs[ex["reference_id"]] if spec["action"] == "resample" and active else None
                    expected_event.update(verified_final_norm_calls={"donor": 1, "recipient": 1} if residual else {},
                        donor_logit_max_abs_error=error,
                        resample_reference_id=source["reference_id"] if source else None,
                        resample_query_token_id=source["final_prompt_token_id"] if source else None,
                        resample_query_position=source["prompt_tokens"] - 1 if source else None)
                if event != expected_event:
                    raise ValueError("Prefix, position, scope, hook or resampling trace mismatch")
        answers = {"clean": ex["clean_gold"], "corrupt": ex["corrupt_gold"], "decoy": ex["unrelated_decoy"]}
        expected_metrics = output_metrics(row["generated_text"], tokens, answers, ex["candidate_token_ids"])
        expected_metrics.update(boundary_metrics(row["generated_text"], ex["clean_gold"], ex["resample_gold"]))
        if any(row[key] != value for key, value in expected_metrics.items()):
            raise ValueError("Generation metric mismatch")
        baseline = lookup[row["case_id"], row["reference"]]
        for metric, delta in zip(DELTA_FIELDS, DELTA_NAMES):
            close(row[delta], row[metric] - baseline[metric], "paired delta")
        if row["condition"].startswith("self_patch_") or row["condition"] == "positive_transfer":
            target = lookup[row["case_id"], "clean"] if row["condition"] == "positive_transfer" else baseline
            assert_self_patch(row, target)
            differences = [abs(a - b) for label in scores for a, b in
                           zip(scores[label]["token_logprobs"], target["candidate_scores"][label]["token_logprobs"])]
            differences += [abs(a - b) for a, b in zip(row["generated_token_logprobs"], target["generated_token_logprobs"])]
            if row["condition"] == "positive_transfer":
                max_positive_error = max(max_positive_error, *differences)
            else:
                max_self_error = max(max_self_error, *differences)
    baseline_counts = []
    for view, mode in VIEW_CORRUPTIONS:
        for label in ("clean", "corrupt"):
            selected = [r for r in rows if (r["view"], r["corruption"], r["condition"]) == (view, mode, label)]
            baseline_counts.append({"view": view, "corruption": mode, "baseline": label, "n": len(selected),
                                    "exact": sum(r[f"exact_{label}"] for r in selected),
                                    "token_budget_stops": sum(r["generation_stop"] == "token_budget" for r in selected)})
    return {"status": "passed", "rows": len(rows), "independent_identities": manifest["n_independent_example_ids"],
            "reference_identities_not_analysis_units": manifest["n_reference_identities"], "cases": len(examples),
            "positive_control_cases": len(examples), "positive_max_logit_error": max_logit_error,
            "positive_max_token_score_error": max_positive_error, "self_patch_max_token_score_error": max_self_error,
            "prefix_position_scope_hook_checks": "passed", "baseline_counts": baseline_counts}


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for variant in ("all", row["variant"]):
            buckets[row["view"], row["corruption"], row["condition"], variant].append(row)
    return {"by_condition": [{"view": view, "corruption": mode, "condition": condition, "variant": variant,
        "n_identities": len(values), **{f"mean_{key}": statistics.fmean(r[key] for r in values) for key in METRICS},
        "token_budget_stops": sum(r["generation_stop"] == "token_budget" for r in values)}
        for (view, mode, condition, variant), values in sorted(buckets.items())]}


def paired_analysis(manifest, rows, draws=4000):
    lookup = {(r["example_id"], r["view"], r["corruption"], r["condition"]): r for r in rows}
    ids = sorted({r["example_id"] for r in rows})
    variants = [lookup[identity, "far_chat", "different_first", "clean"]["variant"] for identity in ids]
    contrasts = []

    def effect(identity, view, mode, action, scope, group, metric):
        controls = [name for name in manifest["groups"] if name.startswith(f"random_{group}_")]
        if not controls:
            raise ValueError("Missing matched controls")
        focal = lookup[identity, view, mode, f"{action}:{scope}:{group}"][metric]
        control = statistics.fmean(lookup[identity, view, mode, f"{action}:{scope}:{name}"][metric] for name in controls)
        return focal - control

    def add(label, metric, values):
        contrasts.append({"contrast": label, "metric": metric, **paired_interval(values, variants, draws)})

    for metric in ("delta_clean_logprob", "delta_exact_clean", "delta_clean_number_content"):
        for group in manifest["protocol"]["distance_groups"]:
            for mode in CORRUPTIONS:
                for scope in SCOPES:
                    values = {view: [effect(identity, view, mode, "patch", scope, group, metric) for identity in ids]
                              for view in DISTANCE_VIEWS}
                    for view, data in values.items():
                        add(f"distance:{view}:{mode}:{scope}:{group}_minus_controls", metric, data)
                    for other in ("near_chat", "short_chat"):
                        add(f"distance:far_chat_minus_{other}:{mode}:{scope}:{group}:control_adjusted_interaction", metric,
                            [a - b for a, b in zip(values["far_chat"], values[other])])
                for view in DISTANCE_VIEWS:
                    add(f"distance:{view}:{mode}:{group}:combined_minus_prompt:control_adjusted", metric,
                        [effect(i, view, mode, "patch", "prompt_and_decode", group, metric)
                         - effect(i, view, mode, "patch", "prompt_only", group, metric) for i in ids])
        for group in manifest["protocol"]["ablation_groups"]:
            for action in ("zero", "resample"):
                values = {view: [effect(i, view, "different_first", action, "prompt_only", group, metric) for i in ids]
                          for view in ABLATION_VIEWS}
                for view, data in values.items():
                    add(f"ablation:{view}:{action}:{group}_minus_controls", metric, data)
                for view in ("near_chat", "short_chat"):
                    add(f"task_ablation:far_chat_minus_{view}:{action}:{group}:control_adjusted_interaction", metric,
                        [a - b for a, b in zip(values["far_chat"], values[view])])
                for view in ("far_raw_space", "far_chat"):
                    add(f"boundary:{view}_minus_far_raw:{action}:{group}:control_adjusted_interaction", metric,
                        [a - b for a, b in zip(values[view], values["far_raw"])])
            for view in ABLATION_VIEWS:
                add(f"ablation:{view}:{group}:resample_minus_zero:control_adjusted", metric,
                    [effect(i, view, "different_first", "resample", "prompt_only", group, metric)
                     - effect(i, view, "different_first", "zero", "prompt_only", group, metric) for i in ids])
    return {"bootstrap_draws": draws, "method": "Paired percentile bootstrap of identities within variant; fixed controls averaged per identity. 95% pointwise intervals.",
            "warning": "Views, corruptions and conditions are repeated measures, not additional identities. Reference donors are not analysis units. No multiplicity correction or interpretation of zero-event intervals as zero population risk.",
            "contrasts": contrasts}


def analyze(directory, draws=4000):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only analyze completed specificity runs")
    if re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"]) is None:
        raise ValueError("Expected a full Git revision")
    verify_committed_sources(manifest["source_sha256"], manifest["git_commit"])
    for name in ("examples", "references", "rows"):
        if hashlib.sha256((directory / f"{name}.jsonl").read_bytes()).hexdigest() != manifest[f"{name}_sha256"]:
            raise ValueError(f"Specificity input hash mismatch: {name}")
    groups = select_groups(manifest["protocol"], manifest["original_protocol"])
    if {key: list(map(tuple, value)) for key, value in manifest["groups"].items()} != groups:
        raise ValueError("Fixed control groups changed")
    if (manifest["model_revision"] != manifest["protocol"]["revision"]
            or manifest["tokenizer_and_chat_template_sha256"] != manifest["protocol"]["tokenizer_and_chat_template_sha256"]):
        raise ValueError("Model or tokenizer revision mismatch")
    examples, references, rows = [read_rows(directory / f"{name}.jsonl") for name in ("examples", "references", "rows")]
    for records, is_reference in ((examples, False), (references, True)):
        if {(r["variant"], r["seed"]) for r in records} != reserved_ids(manifest["protocol"], manifest["arguments"]["stage"], is_reference):
            raise ValueError("Study/reference identities differ from reserved schedule")
    forbidden = prior_ids(ROOT, manifest["protocol"]["protocol"])
    if {(r["variant"], r["seed"]) for r in examples + references} & forbidden:
        raise ValueError("Study/reference identities overlap prior data")
    result = {"validation": validate_rows(manifest, examples, references, rows), "summary": summarize(rows),
              "paired_analysis": paired_analysis(manifest, rows, draws)}
    for name in ("validation", "summary"):
        if (directory / f"{name}.json").exists() and json.loads((directory / f"{name}.json").read_text()) != result[name]:
            raise ValueError(f"Saved specificity {name} does not reproduce")
    return result


def write_analysis(directory, result):
    for name, value in result.items():
        write_json(directory / f"{name}.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    write_analysis(args.run_dir, result)
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
