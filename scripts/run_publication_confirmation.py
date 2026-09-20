#!/usr/bin/env python
"""Run a frozen-head, held-out marker-dependence and generation pilot on SageMaker."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--protocol", default="configs/publication_confirmation.json")
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--n-per-variant", type=int)
    parser.add_argument("--target-tokens", type=int, default=4096)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--random-draws", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true", help="Load the tokenizer, validate and save prompts; do not load model weights")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tokenizer_fingerprint(tokenizer):
    payload = tokenizer.backend_tokenizer.to_str() + json.dumps(tokenizer.chat_template, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_groups(protocol, random_draws):
    from rha.interventions import sample_layer_matched_disjoint
    groups = {name: [tuple(pair) for pair in pairs] for name, pairs in protocol["groups"].items()}
    required = {"address_L22H7", "support_L20H7", "answer_address", "non_address_core"}
    if set(groups) != required:
        raise ValueError("Protocol must specify the four named focal groups")
    for pairs in groups.values():
        if not pairs or len(pairs) != len(set(pairs)):
            raise ValueError("Focal groups must contain distinct, nonempty head lists")
        if any(not 0 <= layer < protocol["num_layers"] or not 0 <= head < protocol["num_heads"] for layer, head in pairs):
            raise ValueError("Protocol includes an out-of-range head")
    if random_draws < 1 or random_draws > 20:
        raise ValueError("Use 1-20 fixed random draws per focal group")
    for index, (name, pairs) in enumerate(list(groups.items())):
        for draw in range(random_draws):
            groups[f"random_{name}_{draw}"] = sample_layer_matched_disjoint(
                pairs, protocol["num_heads"],
                protocol["random_control_seed"] + index * 1000 + draw, banned=pairs,
            )
    return groups


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["marker_mode"], row["condition"], "all")].append(row)
        buckets[(row["marker_mode"], row["condition"], row["variant"])].append(row)
    metrics = (
        "clean_mean_logprob", "answer_logprob_margin", "first_clean_logprob",
        "first_answer_logprob_margin", "exact_clean", "exact_corrupt", "six_digits_only",
        "sole_number_is_clean", "sole_number_is_corrupt", "delta_clean_logprob",
        "delta_answer_margin", "delta_exact_clean",
    )
    result = []
    for (mode, condition, variant), values in sorted(buckets.items()):
        result.append({
            "marker_mode": mode, "condition": condition, "variant": variant,
            "n": len(values),
            **{f"mean_{name}": statistics.fmean(row[name] for row in values) for name in metrics},
            "token_budget_stops": sum(row["generation_stop"] == "token_budget" for row in values),
        })
    return {
        "n_rows": len(rows), "n_independent_example_ids": len({row["example_id"] for row in rows}),
        "interval_note": "Descriptive pilot summary. Marker modes are paired; condition rows are not independent samples. Use the analysis script for paired intervals.",
        "by_condition": result,
    }


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from rha.config import load_runtime_config, resolve_repo_path
    from rha.confirmation import build_examples, historical_ids

    protocol_path = resolve_repo_path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    cfg = load_runtime_config(str(resolve_repo_path(args.config)))
    if cfg.model_id != protocol["model_id"]:
        raise ValueError("Frozen heads are model-specific; use the protocol's exact model")
    n = args.n_per_variant if args.n_per_variant is not None else (2 if args.stage == "pilot" else 32)
    if not 1 <= n <= 1000:
        raise ValueError("n-per-variant must be in [1, 1000] to preserve split separation")
    if args.chunk_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Chunk size and generation budget must be positive")
    seed_base = protocol[f"{args.stage}_seed_base"]
    groups = make_groups(protocol, args.random_draws)
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=False)
    examples = build_examples(
        tokenizer, n, args.target_tokens, args.needle_frac, seed_base,
        args.prompt_format, args.max_len, historical_ids(ROOT / "artifacts_phase2"),
    )
    with (out / "examples.jsonl").open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example, allow_nan=False) + "\n")
    source_files = [Path(__file__), ROOT / "rha/confirmation.py", ROOT / "rha/confirmation_scoring.py",
                    ROOT / "rha/prompts.py", ROOT / "rha/generation.py", ROOT / "rha/interventions.py",
                    ROOT / "rha/modeling.py", ROOT / "rha/config.py", protocol_path]
    manifest = {
        "status": "prepared", "protocol": protocol, "groups": groups,
        "runtime_config": asdict(cfg),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_seed_base": seed_base, "n_per_variant": n,
        "n_example_pairs": len(examples), "n_independent_example_ids": n * 5,
        "prompt_note": "Legacy templates/filler held fixed across marker modes; unmarked prompts are shorter by the marker tokens. This is not new-template validation.",
        "intervention_scope": "final_prompt_token_only",
        "score_note": "Float32 log-softmax; legacy scripts use the logits' native dtype. Report exact-match and candidate extraction separately.",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "numpy")},
        "source_sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "examples_sha256": hashlib.sha256((out / "examples.jsonl").read_bytes()).hexdigest(),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_and_chat_template_sha256": tokenizer_fingerprint(tokenizer),
    }
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(examples)} prompt pairs from {n * 5} identities in {out}", flush=True)
    if args.prepare_only:
        return

    import torch
    from tqdm.auto import tqdm
    from rha.confirmation_scoring import evaluate
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer
    from rha.seed import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Run GPU experiments on SageMaker. Locally use --prepare-only or the unit tests.")
    set_seed(seed_base)
    start = time.monotonic()
    model, model_tokenizer = load_model_and_tokenizer(cfg, trust_remote_code=False)
    if tokenizer_fingerprint(model_tokenizer) != tokenizer_fingerprint(tokenizer):
        raise RuntimeError("Tokenizer changed between prompt preparation and model loading")
    layout = inspect_model_layout(model)
    if len(layout.layers) != protocol["num_layers"] or layout.num_heads != protocol["num_heads"]:
        raise ValueError("Model layout does not match the frozen head protocol")
    manifest.update({"status": "running", "device": torch.cuda.get_device_name(),
                     "dtype": str(next(model.parameters()).dtype),
                     "cuda_version": torch.version.cuda,
                     "model_revision": getattr(model.config, "_commit_hash", None)})
    write_json(out / "manifest.json", manifest)
    selected = sorted({pair for pairs in groups.values() for pair in pairs})
    rows = []
    torch.cuda.reset_peak_memory_stats()

    try:
        with (out / "rows.jsonl").open("w") as handle:
            for example in tqdm(examples, desc="publication confirmation"):
                def run(prompt, **kwargs):
                    return evaluate(model, tokenizer, layout, prompt, example["clean_gold"], example["corrupt_gold"],
                                    args.max_len, args.chunk_size, args.max_new_tokens, **kwargs)

                clean, clean_cache = run(example["clean_prompt"], capture=True)
                corrupt, corrupt_cache = run(example["corrupt_prompt"], capture=True)

                def record(condition, values, baseline, reference):
                    row = {key: value for key, value in example.items() if key not in ("clean_prompt", "corrupt_prompt")}
                    row.update({"condition": condition, "reference": reference, **values,
                                "delta_clean_logprob": values["clean_mean_logprob"] - baseline["clean_mean_logprob"],
                                "delta_answer_margin": values["answer_logprob_margin"] - baseline["answer_logprob_margin"],
                                "delta_exact_clean": values["exact_clean"] - baseline["exact_clean"]})
                    if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                        raise RuntimeError("Non-finite result")
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    rows.append(row)

                record("clean", clean, clean, "clean")
                record("corrupt", corrupt, corrupt, "corrupt")
                for label, baseline, cache in (("clean", clean, clean_cache), ("corrupt", corrupt, corrupt_cache)):
                    identity, _ = run(example[f"{label}_prompt"], pairs=selected, donor=cache)
                    record(f"self_patch_{label}", identity, baseline, label)
                    score_keys = ("clean_mean_logprob", "corrupt_mean_logprob", "first_clean_logprob")
                    if any(abs(identity[key] - baseline[key]) > 1e-5 for key in score_keys) or identity["generated_token_ids"] != baseline["generated_token_ids"]:
                        raise RuntimeError(f"Identity patch changed {label} behavior; inspect hooks/cache before interpreting effects")
                for name, pairs in groups.items():
                    ablated, _ = run(example["clean_prompt"], pairs=pairs)
                    record(f"ablate:{name}", ablated, clean, "clean")
                    patched, _ = run(example["corrupt_prompt"], pairs=pairs, donor=clean_cache)
                    record(f"patch:{name}", patched, corrupt, "corrupt")
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "rows_written": len(rows)})
        write_json(out / "manifest.json", manifest)
        raise
    summary = summarize(rows)
    write_json(out / "summary.json", summary)
    manifest.update({"status": "complete", "rows_written": len(rows),
                     "elapsed_seconds_including_model_load": time.monotonic() - start,
                     "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()})
    write_json(out / "manifest.json", manifest)
    print(f"Completed {len(rows)} condition rows. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
