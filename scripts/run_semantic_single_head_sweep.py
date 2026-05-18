#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_head_pairs(path: Path, n_heads: int) -> List[Tuple[int, int]]:
    pairs = []
    seen = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = (int(row["layer"]), int(row["head"]))
            if pair in seen:
                continue
            pairs.append(pair)
            seen.add(pair)
            if len(pairs) >= n_heads:
                break
    if not pairs:
        raise ValueError(f"No head pairs found in {path}")
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single-head semantic ablation sweep over literal and semantic retrieval prompts."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--heads-csv", default="artifacts_new/day5_patch_single_head.csv")
    parser.add_argument("--out", default="artifacts_phase2/semantic_single_head_sweep.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_single_head_sweep_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=24)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--intervention-scope", choices=["all", "query"], default="query")
    return parser.parse_args()


def summarize(rows, variants, pairs):
    import numpy as np

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n_rows": len(rows),
        "variants": variants,
        "heads": pairs,
        "by_head": [],
        "by_variant": {},
    }

    for layer, head in pairs:
        subset = [row for row in rows if row["layer"] == layer and row["head"] == head]
        item = {
            "layer": layer,
            "head": head,
            "n": len(subset),
            "mean_delta_logprob": mean([row["delta_logprob"] for row in subset]),
            "mean_baseline_logprob": mean([row["baseline_logprob"] for row in subset]),
            "mean_ablated_logprob": mean([row["ablated_logprob"] for row in subset]),
            "by_variant": {},
        }
        for variant in variants:
            variant_subset = [row for row in subset if row["variant"] == variant]
            item["by_variant"][variant] = {
                "n": len(variant_subset),
                "mean_delta_logprob": mean([row["delta_logprob"] for row in variant_subset]),
            }
        summary["by_head"].append(item)

    summary["by_head"].sort(key=lambda row: row["mean_delta_logprob"])

    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        summary["by_variant"][variant] = {
            "n": len(subset),
            "mean_delta_logprob": mean([row["delta_logprob"] for row in subset]),
        }

    return summary


def main() -> None:
    args = parse_args()

    from tqdm.auto import tqdm

    from rha.config import load_runtime_config, resolve_repo_path
    from rha.generation import mean_gold_logprob
    from rha.interventions import HeadAblator, spec_from_pairs
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer
    from rha.prompts import build_dataset, prompt_token_len
    from rha.seed import set_seed

    if args.n_heads < 1:
        raise ValueError("--n-heads must be at least 1.")

    set_seed(args.seed)

    cfg = load_runtime_config(args.config)
    model, tokenizer = load_model_and_tokenizer(cfg)
    layout = inspect_model_layout(model)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    examples = build_dataset(
        tokenizer=tokenizer,
        variants=variants,
        n_per_variant=args.n_per_variant,
        target_tokens=args.target_tokens,
        needle_frac=args.needle_frac,
        seed_base=args.seed,
    )

    pairs = read_head_pairs(resolve_repo_path(args.heads_csv), args.n_heads)
    rows = []

    for ex in tqdm(examples, desc="single-head semantic sweep"):
        baseline_lp = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            gold=ex["gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        actual_tokens = prompt_token_len(tokenizer, ex["prompt"])
        for layer, head in pairs:
            spec = spec_from_pairs([(layer, head)])
            ablated_lp = mean_gold_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["prompt"],
                gold=ex["gold"],
                max_len=args.max_len,
                chunk_size=args.chunk_size,
                intervention=HeadAblator(layout, spec),
                intervention_scope=args.intervention_scope,
            )
            rows.append(
                {
                    "variant": ex["variant"],
                    "seed": ex["seed"],
                    "target_tokens": ex["target_tokens"],
                    "actual_tokens": actual_tokens,
                    "needle_frac": ex["needle_frac"],
                    "gold": ex["gold"],
                    "layer": layer,
                    "head": head,
                    "baseline_logprob": baseline_lp,
                    "ablated_logprob": ablated_lp,
                    "delta_logprob": ablated_lp - baseline_lp,
                }
            )

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, variants, pairs)
    summary.update(
        {
            "model_id": cfg.model_id,
            "n_examples": len(examples),
            "n_heads": len(pairs),
            "heads_csv": str(resolve_repo_path(args.heads_csv)),
            "intervention_scope": args.intervention_scope,
        }
    )

    summary_path = resolve_repo_path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote rows: {out_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
