#!/usr/bin/env python
"""Audit four-fact traces and compare paired, control-adjusted task effects."""

import argparse
import gzip
import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.multifact import (CORRUPTIONS, LABELS, SCOPES, VIEWS, body_parts, candidate_answers, condition_specs,
                           digest, extra_metrics, identity_spec, prior_ids, reserved_ids, select_groups)
from rha.multifact_scoring import assert_control
from rha.temporal import output_metrics
from rha.temporal_scoring import patch_active
from scripts.analyze_publication_confirmation import paired_interval
from scripts.analyze_publication_temporal import close, finite, read_rows
from scripts.run_publication_confirmation import write_json
from scripts.run_publication_multifact import DELTA_FIELDS, DELTA_NAMES
from scripts.run_publication_temporal import verify_committed_sources

METRICS = tuple(dict.fromkeys((*DELTA_FIELDS, *DELTA_NAMES, "exact_corrupt", "exact_other_fact", "six_digits_only",
    "verbatim_clean", "contains_clean_number", "first_number_is_other_fact", "exact_resample",
    "resample_number_content", "contains_reference_number", "first_token_is_clean")))


def validate_design(manifest, examples, references):
    protocol, stage = manifest["protocol"], manifest["arguments"]["stage"]
    identities = {e["example_id"] for e in examples}
    by_case = {e["case_id"]: e for e in examples}
    refs = {r["reference_id"]: r for r in references}
    if (not identities or len(by_case) != len(examples) or len(refs) != len(references)
            or len(identities) != manifest["n_independent_example_ids"] or len(examples) != manifest["n_example_pairs"]
            or len(references) != manifest["n_reference_prompts"]
            or len({(r["variant"], r["seed"]) for r in references}) != manifest["n_reference_identities"]
            or {(e["example_id"], e["view"], e["corruption"]) for e in examples}
            != {(identity, view, mode) for identity in identities for view in VIEWS for mode in CORRUPTIONS}):
        raise ValueError("Incomplete or duplicate multi-fact example/reference grid")
    wrappers = set()

    def check_render(prompt, body):
        if prompt.count(body) != 1:
            raise ValueError("Prompt does not contain the reconstructed task body")
        prefix, suffix = prompt.split(body)
        if not prefix or not suffix:
            raise ValueError("Missing chat wrapper")
        wrappers.add((prefix, suffix))

    def check_spans(record, key):
        sources = record[key]
        if set(sources) != set(record["codes"]):
            raise ValueError("Missing source spans")
        positions = []
        for span in sources.values():
            if (span["end"] != span["start"] + 6
                    or not 0 <= span["start"] < span["end"] < record["question_token_start"] < record["prompt_tokens"] <= protocol["max_len"]
                    or span["distance_to_question"] != record["question_token_start"] - span["end"]):
                raise ValueError("Invalid source/query distance metadata")
            positions.extend(range(span["start"], span["end"]))
        if len(set(positions)) != 24:
            raise ValueError("Overlapping fact-code spans")

    for ex in examples:
        identity, view, mode = ex["example_id"], ex["view"], ex["corruption"]
        spec = identity_spec(protocol, stage, ex["variant"], ex["seed"])
        if (any(ex[k] != v for k, v in spec.items()) or identity != f"{ex['variant']}:{ex['seed']}"
                or ex["case_id"] != f"{identity}/{view}/{mode}" or ex["paragraphs"] < 1
                or ex["target_tokens"] != protocol["target_tokens"]):
            raise ValueError("Incorrect identity/task metadata")
        answers = candidate_answers(spec)
        if (ex["candidate_answers"] != answers or ex["clean_gold"] != answers["clean"] or ex["corrupt_gold"] != answers["corrupt"]
                or ex["other_fact_codes"] != [answers["decoy"], answers["competitor_2"]]):
            raise ValueError("Invalid candidate/answer mapping")
        for label, corruption in (("clean", None), ("corrupt", mode)):
            prompt = ex[f"{label}_prompt"]
            check_render(prompt, body_parts(spec, ex["paragraphs"], view, corruption)[0])
            if digest(prompt) != ex[f"{label}_prompt_sha256"] or any(prompt.count(code) != 1 for code in answers.values()):
                raise ValueError("Prompt hash or four-fact inventory mismatch")
        ids = ex["candidate_token_ids"]
        if (set(ids) != set(LABELS) or any(len(tokens) != 6 or any(not isinstance(t, int) or t < 0 for t in tokens) for tokens in ids.values())
                or len({tokens[0] for tokens in ids.values()}) != 4 or ex["shared_prefix_tokens"] != 0):
            raise ValueError("Invalid six-digit candidate tokens")
        check_spans(ex, "source_spans")
        check_spans(ex, "corrupt_source_spans")
        changed = ex["changed_token_positions"]
        if not changed or changed != sorted(set(changed)):
            raise ValueError("Invalid changed-token positions")
        if mode == "query_selection":
            if ex["source_spans"] != ex["corrupt_source_spans"] or not ex["question_token_start"] <= min(changed) <= max(changed) < ex["prompt_tokens"] - 1:
                raise ValueError("Query corruption changed document positions")
        else:
            expected_changes = []
            for left, right in (("clean", "corrupt"), ("corrupt", "clean")):
                start = ex["source_spans"][answers[left]]["start"]
                expected_changes.extend(start + i for i, (a, b) in enumerate(zip(ids[left], ids[right])) if a != b)
                if ex["source_spans"][answers[left]] != ex["corrupt_source_spans"][answers[right]]:
                    raise ValueError("Value-swap source spans do not exchange codes")
            if changed != sorted(expected_changes) or any(ex["source_spans"][answers[k]] != ex["corrupt_source_spans"][answers[k]] for k in LABELS[2:]):
                raise ValueError("Value swap changed non-answer tokens")
        if mode == "query_selection":
            ref = refs.get(ex["reference_id"])
            if (ref is None or ref["paired_example_id"] != identity or ref["view"] != view or ref["variant"] != ex["variant"]
                    or ref["seed"] != ex["seed"] + protocol["reference_seed_offset"] or ref["gold"] != ex["resample_gold"]
                    or set(ref["codes"]) & set(ex["codes"])):
                raise ValueError("Incorrect task-matched resampling reference")
        elif ex["reference_id"] is not None or ex["resample_gold"] is not None:
            raise ValueError("Unexpected resampling reference")
    if {e["reference_id"] for e in examples if e["reference_id"]} != set(refs):
        raise ValueError("Missing or unused reference prompts")
    for ref in references:
        spec = identity_spec(protocol, stage, ref["variant"], ref["seed"], True)
        if (any(ref[k] != v for k, v in spec.items()) or ref["reference_id"] != f"{ref['variant']}:{ref['seed']}/{ref['view']}"
                or ref["gold"] != spec["codes"][spec["target_slot"]] or digest(ref["prompt"]) != ref["prompt_sha256"]):
            raise ValueError("Invalid reference provenance")
        check_render(ref["prompt"], body_parts(spec, ref["paragraphs"], ref["view"])[0])
        check_spans(ref, "source_spans")
    if len(wrappers) != 1:
        raise ValueError("Chat wrappers differ across views or references")
    for identity in identities:
        pair = {(e["view"], e["corruption"]): e for e in examples if e["example_id"] == identity}
        first = next(iter(pair.values()))
        for key in ("paragraphs", "candidate_answers", "candidate_token_ids", "prompt_tokens", "final_prompt_token_id", "question_token_start"):
            if any(e[key] != first[key] for e in pair.values()):
                raise ValueError("Cases differ beyond the paired manipulations")
        for view in VIEWS:
            if pair[view, "value_swap"]["clean_prompt"] != pair[view, "query_selection"]["clean_prompt"]:
                raise ValueError("Corruptions do not share the same clean prompt")
        for mode in CORRUPTIONS:
            for key in ("source_spans", "corrupt_source_spans"):
                far, near = pair["far_chat", mode], pair["near_chat", mode]
                if any(far[key][code]["distance_to_question"] <= near[key][code]["distance_to_question"] for code in first["codes"]):
                    raise ValueError("Invalid far/near source distance")


def validate_rows(manifest, examples, references, rows):
    if not finite(rows):
        raise ValueError("Non-finite multi-fact readout")
    validate_design(manifest, examples, references)
    by_case, refs = {e["case_id"]: e for e in examples}, {r["reference_id"]: r for r in references}
    specs = {e["case_id"]: condition_specs(e, manifest["groups"], manifest["protocol"]) for e in examples}
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    expected = {(case, name) for case, conditions in specs.items() for name in conditions}
    if len(lookup) != len(rows) or set(lookup) != expected or len(rows) != manifest["rows_written"] or len(rows) != manifest["expected_rows"]:
        raise ValueError("Incomplete or duplicate multi-fact condition grid")
    max_self_error, max_positive_error, max_logit_error = 0.0, 0.0, 0.0
    for row in rows:
        ex, spec = by_case[row["case_id"]], specs[row["case_id"]][row["condition"]]
        if any(row[key] != value for key, value in ex.items() if key not in ("clean_prompt", "corrupt_prompt")):
            raise ValueError("Row/example metadata mismatch")
        if row["reference"] != spec["recipient"]:
            raise ValueError("Incorrect reference baseline")
        scores = row["candidate_scores"]
        if set(scores) != set(LABELS):
            raise ValueError("Missing four-candidate scores")
        for label, score in scores.items():
            if score["token_ids"] != ex["candidate_token_ids"][label] or len(score["token_logprobs"]) != 6 or score["stop"] != "teacher_forced":
                raise ValueError("Invalid candidate scoring branch")
            close(score["sum_logprob"], sum(score["token_logprobs"]), "token sum")
            close(score["mean_logprob"], score["sum_logprob"] / 6, "token mean")
            close(row[f"{label}_mean_logprob"], score["mean_logprob"], "candidate mean")
        close(row["answer_logprob_margin"], scores["clean"]["sum_logprob"] - scores["corrupt"]["sum_logprob"], "answer margin")
        close(row["clean_decoy_logprob_margin"], scores["clean"]["sum_logprob"] - scores["decoy"]["sum_logprob"], "other fact margin")
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
        branches = [(s["token_ids"], s["steps"]) for s in scores.values()] + [(tokens, row["generation_steps"])]
        for branch_tokens, steps in branches:
            if len(steps) != len(branch_tokens):
                raise ValueError("Missing step trace")
            for index, event in enumerate(steps):
                token = ex["final_prompt_token_id"] if index == 0 else branch_tokens[index - 1]
                position, active = ex["prompt_tokens"] - 1 + index, patch_active(spec["scope"], index)
                donor = spec["donor"] in ("clean", "corrupt")
                expected_event = {"step": index, "recipient_input_token_id": token,
                    "donor_input_token_id": token if donor else None,
                    "recipient_cache_length_before": position, "donor_cache_length_before": position if donor else None,
                    "patch_applied": active, "verified_hook_calls": {str(layer): int(active) for layer, _ in spec["pairs"]}}
                if spec["action"] in ("zero", "resample", "residual"):
                    error, residual = event.get("donor_logit_max_abs_error"), spec["action"] == "residual"
                    if residual:
                        if not isinstance(error, (int, float)) or not 0 <= error <= 1e-5:
                            raise ValueError("Positive transfer logit check failed")
                        max_logit_error = max(max_logit_error, error)
                    elif error is not None:
                        raise ValueError("Unexpected donor logit comparison")
                    source = refs[ex["reference_id"]] if spec["action"] == "resample" and active else None
                    expected_event.update(verified_final_norm_calls={"donor": 1, "recipient": 1} if residual else {},
                        donor_logit_max_abs_error=error, resample_reference_id=source["reference_id"] if source else None,
                        resample_query_token_id=source["final_prompt_token_id"] if source else None,
                        resample_query_position=source["prompt_tokens"] - 1 if source else None)
                if event != expected_event:
                    raise ValueError("Prefix, position, scope, hook or resampling trace mismatch")
        expected_metrics = output_metrics(row["generated_text"], tokens, ex["candidate_answers"], ex["candidate_token_ids"])
        expected_metrics.update(extra_metrics(row["generated_text"], scores, ex["candidate_answers"], ex["resample_gold"]))
        if any(row[key] != value for key, value in expected_metrics.items()):
            raise ValueError("Generation/candidate metric mismatch")
        baseline = lookup[row["case_id"], row["reference"]]
        for metric, delta in zip(DELTA_FIELDS, DELTA_NAMES):
            close(row[delta], row[metric] - baseline[metric], "paired delta")
        if row["condition"].startswith("self_patch_") or row["condition"] == "positive_transfer":
            target = lookup[row["case_id"], "clean"] if row["condition"] == "positive_transfer" else baseline
            assert_control(row, target)
            differences = [abs(a - b) for label in LABELS for a, b in zip(scores[label]["token_logprobs"], target["candidate_scores"][label]["token_logprobs"])]
            differences += [abs(a - b) for a, b in zip(row["generated_token_logprobs"], target["generated_token_logprobs"])]
            if row["condition"] == "positive_transfer":
                max_positive_error = max(max_positive_error, *differences)
            else:
                max_self_error = max(max_self_error, *differences)
    for ex in examples:
        if ex["corruption"] == "query_selection":
            other = f"{ex['example_id']}/{ex['view']}/value_swap"
            assert_control(lookup[ex["case_id"], "clean"], lookup[other, "clean"])
    baseline_counts = []
    for view in VIEWS:
        for mode in CORRUPTIONS:
            for label in ("clean", "corrupt"):
                selected = [r for r in rows if (r["view"], r["corruption"], r["condition"]) == (view, mode, label)]
                baseline_counts.append({"view": view, "corruption": mode, "baseline": label, "n": len(selected),
                    "exact": sum(r[f"exact_{label}"] for r in selected),
                    "ranks_correct_fact_first": sum(r["candidate_scores"][label]["sum_logprob"] > max(s["sum_logprob"] for k, s in r["candidate_scores"].items() if k != label) for r in selected),
                    "token_budget_stops": sum(r["generation_stop"] == "token_budget" for r in selected)})
    return {"status": "passed", "rows": len(rows), "independent_identities": manifest["n_independent_example_ids"],
        "reference_identities_not_analysis_units": manifest["n_reference_identities"], "cases": len(examples),
        "positive_control_cases": len(examples), "positive_max_logit_error": max_logit_error,
        "positive_max_token_score_error": max_positive_error, "self_patch_max_token_score_error": max_self_error,
        "four_candidate_prefix_position_scope_hook_checks": "passed", "baseline_counts": baseline_counts}


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for family in ("all", row["variant"]):
            buckets[row["view"], row["corruption"], row["condition"], family].append(row)
    return {"by_condition": [{"view": view, "corruption": mode, "condition": condition, "family": family,
        "n_identities": len(values), **{f"mean_{key}": statistics.fmean(r[key] for r in values) for key in METRICS},
        "token_budget_stops": sum(r["generation_stop"] == "token_budget" for r in values)}
        for (view, mode, condition, family), values in sorted(buckets.items())]}


def paired_analysis(manifest, rows, draws=4000):
    lookup = {(r["example_id"], r["view"], r["corruption"], r["condition"]): r for r in rows}
    identities = sorted({r["example_id"] for r in rows})
    families = [lookup[i, "far_chat", "value_swap", "clean"]["variant"] for i in identities]
    contrasts = []

    def effect(identity, view, mode, action, scope, group, metric):
        controls = [name for name in manifest["groups"] if name.startswith(f"random_{group}_")]
        if not controls:
            raise ValueError("Missing matched controls")
        focal = lookup[identity, view, mode, f"{action}:{scope}:{group}"][metric]
        return focal - statistics.fmean(lookup[identity, view, mode, f"{action}:{scope}:{name}"][metric] for name in controls)

    def add(label, metric, values):
        contrasts.append({"contrast": label, "metric": metric, **paired_interval(values, families, draws)})

    for metric in ("delta_clean_logprob", "delta_best_other_margin", "delta_exact_clean", "delta_clean_number_content"):
        for group in manifest["protocol"]["distance_groups"]:
            values = {(view, mode, scope): [effect(i, view, mode, "patch", scope, group, metric) for i in identities]
                      for view in VIEWS for mode in CORRUPTIONS for scope in SCOPES}
            for (view, mode, scope), data in values.items():
                add(f"patch:{view}:{mode}:{scope}:{group}_minus_controls", metric, data)
            for scope in SCOPES:
                distances = {}
                for mode in CORRUPTIONS:
                    distances[mode] = [a - b for a, b in zip(values["far_chat", mode, scope], values["near_chat", mode, scope])]
                    add(f"distance:far_minus_near:{mode}:{scope}:{group}:control_adjusted", metric, distances[mode])
                for view in VIEWS:
                    add(f"task:query_minus_swap:{view}:{scope}:{group}:control_adjusted", metric,
                        [a - b for a, b in zip(values[view, "query_selection", scope], values[view, "value_swap", scope])])
                add(f"task_distance:query_minus_swap:far_minus_near:{scope}:{group}:control_adjusted", metric,
                    [a - b for a, b in zip(distances["query_selection"], distances["value_swap"])])
            for view in VIEWS:
                for mode in CORRUPTIONS:
                    add(f"scope:combined_minus_prompt:{view}:{mode}:{group}:control_adjusted", metric,
                        [a - b for a, b in zip(values[view, mode, "prompt_and_decode"], values[view, mode, "prompt_only"])])
        for group in manifest["protocol"]["ablation_groups"]:
            values = {(view, action): [effect(i, view, "query_selection", action, "prompt_only", group, metric) for i in identities]
                      for view in VIEWS for action in ("zero", "resample")}
            for (view, action), data in values.items():
                add(f"ablation:{view}:{action}:{group}_minus_controls", metric, data)
            for action in ("zero", "resample"):
                add(f"ablation:far_minus_near:{action}:{group}:control_adjusted", metric,
                    [a - b for a, b in zip(values["far_chat", action], values["near_chat", action])])
            for view in VIEWS:
                add(f"ablation:{view}:resample_minus_zero:{group}:control_adjusted", metric,
                    [a - b for a, b in zip(values[view, "resample"], values[view, "zero"])])
    return {"bootstrap_draws": draws,
        "method": "Paired percentile bootstrap of identities within task family; fixed controls averaged within identity. 95% pointwise intervals.",
        "warning": "Views, corruptions and conditions are repeated measures. Reference donors are not analysis units. No multiplicity correction; zero-event intervals do not establish zero population risk.",
        "contrasts": contrasts}


def read_compressed_rows(directory, manifest):
    path = directory / "rows.jsonl.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["rows_gzip_sha256"]:
        raise ValueError("Multi-fact gzip hash mismatch")
    rows, logical_hash = [], hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for line in handle:
            logical_hash.update(line)
            if line.strip():
                rows.append(json.loads(line))
    if logical_hash.hexdigest() != manifest["rows_sha256"]:
        raise ValueError("Multi-fact logical rows hash mismatch")
    return rows


def analyze(directory, draws=4000):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only analyze completed multi-fact runs")
    if re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"]) is None:
        raise ValueError("Expected a full Git revision")
    verify_committed_sources(manifest["source_sha256"], manifest["git_commit"])
    for key, source_key in (("protocol", "protocol_source"), ("original_protocol", "original_protocol_source")):
        relative = manifest[source_key]
        if relative not in manifest["source_sha256"]:
            raise ValueError("Missing committed protocol hash")
        committed = subprocess.check_output(["git", "show", f"{manifest['git_commit']}:{relative}"], cwd=ROOT)
        if json.loads(committed) != manifest[key]:
            raise ValueError("Manifest protocol differs from committed protocol")
    for name in ("examples", "references"):
        if hashlib.sha256((directory / f"{name}.jsonl").read_bytes()).hexdigest() != manifest[f"{name}_sha256"]:
            raise ValueError(f"Multi-fact input hash mismatch: {name}")
    groups = select_groups(manifest["protocol"], manifest["original_protocol"])
    if {key: list(map(tuple, value)) for key, value in manifest["groups"].items()} != groups:
        raise ValueError("Fixed control groups changed")
    if (manifest["model_revision"] != manifest["protocol"]["revision"]
            or manifest["tokenizer_and_chat_template_sha256"] != manifest["protocol"]["tokenizer_and_chat_template_sha256"]):
        raise ValueError("Model or tokenizer revision mismatch")
    examples, references = [read_rows(directory / f"{name}.jsonl") for name in ("examples", "references")]
    rows = read_compressed_rows(directory, manifest)
    for records, is_reference in ((examples, False), (references, True)):
        if {(r["variant"], r["seed"]) for r in records} != reserved_ids(manifest["protocol"], manifest["arguments"]["stage"], is_reference):
            raise ValueError("Study/reference identities differ from reserved schedule")
    forbidden = prior_ids(ROOT, manifest["protocol"]["protocol"])
    if {r["seed"] for r in examples + references} & {seed for _, seed in forbidden}:
        raise ValueError("Study/reference identities overlap prior data")
    result = {"validation": validate_rows(manifest, examples, references, rows), "summary": summarize(rows),
              "paired_analysis": paired_analysis(manifest, rows, draws)}
    for name in result:
        path = directory / f"{name}.json"
        if path.exists() and json.loads(path.read_text()) != result[name]:
            raise ValueError(f"Saved multi-fact {name} does not reproduce")
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
