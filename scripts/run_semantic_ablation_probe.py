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


def read_head_pairs(path: Path, top_k: int) -> List[Tuple[int, int]]:
    pairs = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((int(row["layer"]), int(row["head"])))
            if len(pairs) >= top_k:
                break
    if not pairs:
        raise ValueError(f"No head pairs found in {path}")
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and head-ablated log-probability on literal and semantic retrieval prompts."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--heads-csv", default="artifacts_new/day5_patch_single_head.csv")
    parser.add_argument("--out", default="artifacts_phase2/semantic_ablation_probe.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_ablation_probe_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--random-control-seed", type=int, default=2026)
    parser.add_argument("--n-random-draws", type=int, default=1)
    parser.add_argument("--intervention-scope", choices=["all", "query"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from tqdm.auto import tqdm

    from rha.config import load_runtime_config, resolve_repo_path
    from rha.generation import mean_gold_logprob
    from rha.interventions import HeadAblator, sample_layer_matched_disjoint, spec_from_pairs
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer
    from rha.prompts import build_dataset, prompt_token_len
    from rha.seed import set_seed

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

    top_pairs = read_head_pairs(resolve_repo_path(args.heads_csv), args.top_k)
    if args.n_random_draws < 1:
        raise ValueError("--n-random-draws must be at least 1.")

    random_pair_draws = [
        sample_layer_matched_disjoint(
            top_pairs=top_pairs,
            num_heads_per_layer=layout.num_heads,
            seed=args.random_control_seed + draw_idx,
            banned=top_pairs,
        )
        for draw_idx in range(args.n_random_draws)
    ]
    top_spec = spec_from_pairs(top_pairs)
    random_specs = [spec_from_pairs(pairs) for pairs in random_pair_draws]

    rows = []
    for ex in tqdm(examples, desc="semantic ablation probe"):
        base_lp = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            gold=ex["gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        top_lp = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            gold=ex["gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
            intervention=HeadAblator(layout, top_spec),
            intervention_scope=args.intervention_scope,
        )
        rand_lps = []
        for random_spec in random_specs:
            rand_lps.append(
                mean_gold_logprob(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=ex["prompt"],
                    gold=ex["gold"],
                    max_len=args.max_len,
                    chunk_size=args.chunk_size,
                    intervention=HeadAblator(layout, random_spec),
                    intervention_scope=args.intervention_scope,
                )
            )
        rand_arr = np.array(rand_lps, dtype=np.float64)
        rand_delta_arr = rand_arr - base_lp
        rows.append(
            {
                "variant": ex["variant"],
                "seed": ex["seed"],
                "target_tokens": ex["target_tokens"],
                "actual_tokens": prompt_token_len(tokenizer, ex["prompt"]),
                "needle_frac": ex["needle_frac"],
                "gold": ex["gold"],
                "baseline_logprob": base_lp,
                "topk_ablated_logprob": top_lp,
                "randomk_ablated_logprob": float(rand_arr[0]),
                "randomk_ablated_logprob_mean": float(rand_arr.mean()),
                "randomk_ablated_logprob_std": float(rand_arr.std(ddof=0)),
                "delta_topk": top_lp - base_lp,
                "delta_randomk": float(rand_delta_arr[0]),
                "delta_randomk_mean": float(rand_delta_arr.mean()),
                "delta_randomk_std": float(rand_delta_arr.std(ddof=0)),
                "delta_randomk_all": json.dumps(rand_delta_arr.tolist()),
            }
        )

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_id": cfg.model_id,
        "n_examples": len(rows),
        "variants": variants,
        "top_pairs": top_pairs,
        "random_pairs": random_pair_draws[0],
        "random_pair_draws": random_pair_draws,
        "n_random_draws": args.n_random_draws,
        "intervention_scope": args.intervention_scope,
        "by_variant": {},
    }

    def summarize_subset(subset):
        random_draw_mean_deltas = []
        for draw_idx in range(args.n_random_draws):
            random_draw_mean_deltas.append(
                float(
                    np.mean(
                        [
                            json.loads(row["delta_randomk_all"])[draw_idx]
                            for row in subset
                        ]
                    )
                )
            )
        return {
            "n": len(subset),
            "mean_baseline_logprob": float(np.mean([row["baseline_logprob"] for row in subset])),
            "mean_delta_topk": float(np.mean([row["delta_topk"] for row in subset])),
            "mean_delta_randomk": float(np.mean([row["delta_randomk"] for row in subset])),
            "mean_delta_randomk_mean": float(np.mean([row["delta_randomk_mean"] for row in subset])),
            "mean_delta_randomk_std": float(np.mean([row["delta_randomk_std"] for row in subset])),
            "random_draw_mean_deltas": random_draw_mean_deltas,
            "random_draw_mean_delta_mean": float(np.mean(random_draw_mean_deltas)),
            "random_draw_mean_delta_std": float(np.std(random_draw_mean_deltas, ddof=0)),
            "topk_minus_random_draw_mean": float(
                np.mean([row["delta_topk"] for row in subset]) - np.mean(random_draw_mean_deltas)
            ),
        }

    summary["overall"] = summarize_subset(rows)
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        summary["by_variant"][variant] = summarize_subset(subset)

    summary_path = resolve_repo_path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote rows: {out_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
