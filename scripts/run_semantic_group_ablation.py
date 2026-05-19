#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_groups(path: Path, selected_groups: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    groups: Dict[str, List[Tuple[int, int]]] = {}
    selected = set(selected_groups)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"]
            if selected and group not in selected:
                continue
            groups.setdefault(group, [])
            pair = (int(row["layer"]), int(row["head"]))
            if pair not in groups[group]:
                groups[group].append(pair)
    if not groups:
        raise ValueError(f"No groups found in {path}.")
    return groups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run query-step group ablations for semantic retrieval circuit components."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--groups-csv", default="configs/semantic_functional_groups.csv")
    parser.add_argument("--groups", default="answer_address,non_address_core,strong13,core19,first_token_sink,query_tail")
    parser.add_argument("--out", default="artifacts_phase2/semantic_group_ablation.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_group_ablation_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--intervention-scope", choices=["all", "query"], default="query")
    return parser.parse_args()


def summarize(rows, variants, groups):
    import numpy as np

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n_rows": len(rows),
        "variants": variants,
        "groups": {name: pairs for name, pairs in groups.items()},
        "by_group": {},
    }

    for group, pairs in groups.items():
        subset = [row for row in rows if row["group"] == group]
        item = {
            "n": len(subset),
            "n_heads": len(pairs),
            "heads": pairs,
            "mean_baseline_logprob": mean([row["baseline_logprob"] for row in subset]),
            "mean_ablated_logprob": mean([row["ablated_logprob"] for row in subset]),
            "mean_delta_logprob": mean([row["delta_logprob"] for row in subset]),
            "negative_examples": int(sum(1 for row in subset if row["delta_logprob"] < 0)),
            "by_variant": {},
        }
        for variant in variants:
            variant_subset = [row for row in subset if row["variant"] == variant]
            item["by_variant"][variant] = {
                "n": len(variant_subset),
                "mean_delta_logprob": mean([row["delta_logprob"] for row in variant_subset]),
                "negative_examples": int(sum(1 for row in variant_subset if row["delta_logprob"] < 0)),
            }
        summary["by_group"][group] = item

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

    selected_groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    groups = read_groups(resolve_repo_path(args.groups_csv), selected_groups)

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

    rows = []
    for ex in tqdm(examples, desc="semantic group ablation"):
        baseline_lp = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            gold=ex["gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        actual_tokens = prompt_token_len(tokenizer, ex["prompt"])
        for group, pairs in groups.items():
            ablated_lp = mean_gold_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["prompt"],
                gold=ex["gold"],
                max_len=args.max_len,
                chunk_size=args.chunk_size,
                intervention=HeadAblator(layout, spec_from_pairs(pairs)),
                intervention_scope=args.intervention_scope,
            )
            rows.append(
                {
                    "group": group,
                    "variant": ex["variant"],
                    "seed": ex["seed"],
                    "target_tokens": ex["target_tokens"],
                    "actual_tokens": actual_tokens,
                    "needle_frac": ex["needle_frac"],
                    "gold": ex["gold"],
                    "n_heads": len(pairs),
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

    summary = summarize(rows, variants, groups)
    summary.update(
        {
            "model_id": cfg.model_id,
            "n_examples": len(examples),
            "groups_csv": str(resolve_repo_path(args.groups_csv)),
            "intervention_scope": args.intervention_scope,
            "target_tokens": args.target_tokens,
            "needle_frac": args.needle_frac,
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
