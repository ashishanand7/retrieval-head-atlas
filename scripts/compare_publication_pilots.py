#!/usr/bin/env python
"""Audit and compare paired raw/chat pilot artifacts without rerunning the model."""

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.confirmation import generation_metrics, historical_ids
from scripts.analyze_publication_confirmation import analyze, paired_interval
from scripts.run_publication_confirmation import summarize


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_diagnostics(text, answer):
    # Post-hoc diagnostics: unlike the frozen metric, repeated correct numbers are allowed.
    numbers = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    return {
        "first_number_matches": bool(numbers and numbers[0] == answer),
        "only_distinct_number_matches": bool(numbers and set(numbers) == {answer}),
    }


def load_checked(directory, root=ROOT):
    manifest = json.loads((directory / "manifest.json").read_text())
    paired = analyze(directory, 4000)  # Also checks the complete, nonduplicated condition grid.
    rows = read_jsonl(directory / "rows.jsonl")
    examples = read_jsonl(directory / "examples.jsonl")
    if sha256(directory / "examples.jsonl") != manifest["examples_sha256"]:
        raise ValueError("Example file hash mismatch")
    revision = manifest["git_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a full recorded Git commit ID")
    for relative, expected in manifest["source_sha256"].items():
        recorded = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root)
        if hashlib.sha256(recorded).hexdigest() != expected:
            raise ValueError(f"Recorded experiment source differs from its Git revision: {relative}")
    if summarize(rows) != json.loads((directory / "summary.json").read_text()):
        raise ValueError("Saved summary does not reproduce from rows")
    by_example = {(row["example_id"], row["marker_mode"]): row for row in examples}
    if len(by_example) != len(examples) or len(examples) != manifest["n_example_pairs"]:
        raise ValueError("Duplicate or missing saved examples")
    history = historical_ids(root / "artifacts_phase2")
    if any((row["variant"], row["seed"]) in history for row in examples):
        raise ValueError("Pilot overlaps historical example identities")
    for example in examples:
        for label in ("clean", "corrupt"):
            if hashlib.sha256(example[f"{label}_prompt"].encode()).hexdigest() != example[f"{label}_prompt_sha256"]:
                raise ValueError("Individual prompt hash mismatch")
        if example["prompt_tokens"] > manifest["arguments"]["max_len"]:
            raise ValueError("Overlength saved prompt")
    lookup = {(row["example_id"], row["marker_mode"], row["condition"]): row for row in rows}
    self_patch_max_error = 0.0
    for row in rows:
        example = by_example[row["example_id"], row["marker_mode"]]
        for key, value in example.items():
            if key not in ("clean_prompt", "corrupt_prompt") and row[key] != value:
                raise ValueError(f"Row/example mismatch: {key}")
        for key, expected in generation_metrics(row["generated_text"], row["clean_gold"], row["corrupt_gold"]).items():
            if row[key] != expected:
                raise ValueError(f"Generation metric mismatch: {key}")
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite measurement")
        condition = row["condition"]
        expected_reference = "clean" if condition == "clean" or condition == "self_patch_clean" or condition.startswith("ablate:") else "corrupt"
        if row["reference"] != expected_reference:
            raise ValueError("Incorrect intervention reference")
        baseline = lookup[row["example_id"], row["marker_mode"], row["reference"]]
        for field, delta in (("clean_mean_logprob", "delta_clean_logprob"),
                             ("answer_logprob_margin", "delta_answer_margin"),
                             ("exact_clean", "delta_exact_clean")):
            if not math.isclose(row[field] - baseline[field], row[delta], abs_tol=1e-10):
                raise ValueError(f"Paired delta mismatch: {delta}")
        if condition.startswith("self_patch"):
            fields = ("clean_mean_logprob", "corrupt_mean_logprob", "first_clean_logprob",
                      "answer_logprob_margin", "first_answer_logprob_margin")
            error = max(abs(row[key] - baseline[key]) for key in fields)
            self_patch_max_error = max(error, self_patch_max_error)
            if error > 1e-5 or row["generated_token_ids"] != baseline["generated_token_ids"]:
                raise ValueError("Self patch changed scores or generated tokens")
    return manifest, rows, examples, lookup, paired, self_patch_max_error


def compare(raw_dir, chat_dir, root=ROOT):
    loaded = {fmt: load_checked(path, root) for fmt, path in (("raw", raw_dir), ("chat", chat_dir))}
    raw_manifest, chat_manifest = loaded["raw"][0], loaded["chat"][0]
    if raw_manifest["arguments"]["prompt_format"] != "raw" or chat_manifest["arguments"]["prompt_format"] != "chat":
        raise ValueError("Expected raw and chat runs in that order")
    for key in ("protocol", "groups", "runtime_config", "resolved_seed_base", "n_per_variant", "packages",
                "source_sha256", "model_revision", "tokenizer_and_chat_template_sha256", "dtype"):
        if raw_manifest[key] != chat_manifest[key]:
            raise ValueError(f"Runs differ beyond prompt format: {key}")
    for key, value in raw_manifest["arguments"].items():
        if key not in ("prompt_format", "out_dir") and value != chat_manifest["arguments"][key]:
            raise ValueError(f"Run argument mismatch: {key}")
    raw_examples = {(row["example_id"], row["marker_mode"]): row for row in loaded["raw"][2]}
    chat_examples = {(row["example_id"], row["marker_mode"]): row for row in loaded["chat"][2]}
    if raw_examples.keys() != chat_examples.keys():
        raise ValueError("Formats are not example-paired")
    for key, example in raw_examples.items():
        chat = chat_examples[key]
        for field in ("clean_gold", "corrupt_gold", "seed", "variant", "filler_words", "answer_tokens"):
            if example[field] != chat[field]:
                raise ValueError(f"Pairing mismatch: {field}")
        if any(example[f"{label}_prompt"] not in chat[f"{label}_prompt"] for label in ("clean", "corrupt")):
            raise ValueError("Chat prompt does not preserve the raw document and question")

    result = {
        "audit": {"status": "passed", "rows_per_format": len(loaded["raw"][1]),
                  "independent_identities_shared_across_formats_and_markers": raw_manifest["n_independent_example_ids"],
                  "historical_identity_overlap": 0,
                  "self_patch_max_score_error": {fmt: value[5] for fmt, value in loaded.items()},
                  "same_model_tokenizer_code_groups_answers_and_filler": True},
        "source_sha256": {f"{fmt}/{name}": sha256(path / name) for fmt, path in (("raw", raw_dir), ("chat", chat_dir))
                          for name in ("manifest.json", "examples.jsonl", "rows.jsonl", "summary.json")},
        "diagnostic_note": "First-number and repeated-identical-number checks are post-hoc content diagnostics, not replacements for the frozen exact-match or sole-number metrics. They inspect the recorded output only, including token-budget-limited text.",
        "interval_note": "Within-identity paired, variant-stratified 95% pointwise bootstrap intervals. Descriptive pilot evidence, not independent replication across formats or multiplicity-adjusted tests. Sample counts are recorded with each estimate.",
        "runs": {}, "conditions": [], "token_decomposition": [], "changed_outputs": [], "chat_minus_raw": [],
    }
    for fmt, (manifest, rows, examples, lookup, _paired, _error) in loaded.items():
        result["runs"][fmt] = {
            "elapsed_minutes_including_model_load": manifest["elapsed_seconds_including_model_load"] / 60,
            "peak_allocated_cuda_gib": manifest["peak_cuda_memory_bytes"] / 1024 ** 3,
            "mean_generated_tokens": statistics.fmean(len(row["generated_token_ids"]) for row in rows),
            "prompt_tokens_min": min(row["prompt_tokens"] for row in examples),
            "prompt_tokens_max": max(row["prompt_tokens"] for row in examples),
        }
        for marker in ("marked", "unmarked"):
            for condition in sorted({row["condition"] for row in rows}):
                selected = [row for row in rows if row["marker_mode"] == marker and row["condition"] == condition]
                by_variant = [row["variant"] for row in selected]
                metrics = ("clean_mean_logprob", "delta_clean_logprob", "delta_answer_margin", "first_clean_logprob", "answer_logprob_margin")
                record = {"format": fmt, "marker": marker, "condition": condition, "n": len(selected),
                          **{f"mean_{field}": statistics.fmean(row[field] for row in selected) for field in metrics}}
                for field in ("exact_clean", "exact_corrupt", "sole_number_is_clean", "sole_number_is_corrupt", "six_digits_only"):
                    record[f"count_{field}"] = sum(row[field] for row in selected)
                for label in ("clean", "corrupt"):
                    for diagnostic in ("first_number_matches", "only_distinct_number_matches"):
                        record[f"count_{label}_{diagnostic}"] = sum(content_diagnostics(row["generated_text"], row[f"{label}_gold"])[diagnostic] for row in selected)
                record["token_budget_stops"] = sum(row["generation_stop"] == "token_budget" for row in selected)
                record["positive_patch_or_ablation_deltas"] = sum(row["delta_clean_logprob"] > 0 for row in selected)
                record["delta_clean_logprob_interval"] = paired_interval([row["delta_clean_logprob"] for row in selected], by_variant)
                result["conditions"].append(record)
                if condition.startswith(("ablate:", "patch:")):
                    total = statistics.fmean(row["delta_clean_logprob"] * row["answer_tokens"] for row in selected)
                    first = statistics.fmean(row["first_clean_logprob"] - lookup[row["example_id"], marker, row["reference"]]["first_clean_logprob"] for row in selected)
                    result["token_decomposition"].append({"format": fmt, "marker": marker, "condition": condition,
                        "mean_sequence_logprob_delta": total, "mean_first_token_delta": first,
                        "mean_remaining_tokens_sum_delta": total - first,
                        "first_over_total_ratio": first / total if abs(total) > 1e-12 else None})
                    for row in selected:
                        baseline = lookup[row["example_id"], marker, row["reference"]]
                        if row["generated_text"] != baseline["generated_text"]:
                            result["changed_outputs"].append({"format": fmt, "marker": marker, "condition": condition,
                                "example_id": row["example_id"], "clean_gold": row["clean_gold"],
                                "corrupt_gold": row["corrupt_gold"], "baseline_text": baseline["generated_text"],
                                "generated_text": row["generated_text"], "stop": row["generation_stop"]})

    identities = sorted({key[0] for key in raw_examples})
    for marker in ("marked", "unmarked"):
        for condition in sorted({row["condition"] for row in loaded["raw"][1]}):
            keys = [(identity, marker, condition) for identity in identities]
            variants = [loaded["raw"][3][key]["variant"] for key in keys]
            for metric in ("delta_clean_logprob", "delta_answer_margin", "exact_clean"):
                differences = [loaded["chat"][3][key][metric] - loaded["raw"][3][key][metric] for key in keys]
                result["chat_minus_raw"].append({"marker": marker, "condition": condition, "metric": metric,
                                                **paired_interval(differences, variants)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "artifacts_publication/pilot_4096_raw")
    parser.add_argument("--chat-dir", type=Path, default=ROOT / "artifacts_publication/pilot_4096_chat")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_publication/pilot_comparison.json")
    args = parser.parse_args()
    result = compare(args.raw_dir, args.chat_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audit passed: {result['audit']['rows_per_format']} rows per format, "
          f"{result['audit']['independent_identities_shared_across_formats_and_markers']} shared independent identities.")
    print(f"Wrote comparison and changed-output audit: {args.out}")


if __name__ == "__main__":
    main()
