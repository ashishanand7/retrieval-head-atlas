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
        description="Measure clean-vs-corrupt activation differences for semantic retrieval components."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--groups-csv", default="configs/semantic_functional_groups.csv")
    parser.add_argument(
        "--groups",
        default="answer_address,non_address_core,strong13,core19,first_token_sink,query_tail,answer_address_inactive_control,query_tail_inactive_control",
    )
    parser.add_argument("--out", default="artifacts_phase2/semantic_activation_delta.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_activation_delta_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-secret-tries", type=int, default=2000)
    parser.add_argument("--include-head-rows", action="store_true")
    return parser.parse_args()


def mean(values):
    import numpy as np

    return float(np.mean(values)) if values else float("nan")


def vector_for_pairs(layout, cache, pairs):
    import torch

    chunks = []
    for layer, head in pairs:
        x = cache[int(layer)]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.detach().float().cpu().reshape(
            x.shape[0],
            x.shape[1],
            layout.num_heads,
            layout.head_dim,
        )
        chunks.append(x[:, :, int(head), :].reshape(-1))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def metrics_for_vectors(clean_vec, corrupt_vec) -> Dict:
    import torch

    diff = clean_vec - corrupt_vec
    clean_norm = float(torch.linalg.vector_norm(clean_vec))
    corrupt_norm = float(torch.linalg.vector_norm(corrupt_vec))
    diff_norm = float(torch.linalg.vector_norm(diff))
    denom = 0.5 * (clean_norm + corrupt_norm)
    cosine = float(torch.nn.functional.cosine_similarity(clean_vec, corrupt_vec, dim=0)) if clean_vec.numel() else float("nan")
    return {
        "clean_l2": clean_norm,
        "corrupt_l2": corrupt_norm,
        "diff_l2": diff_norm,
        "relative_diff_l2": diff_norm / denom if denom > 0 else float("nan"),
        "cosine_similarity": cosine,
        "mean_abs_diff": float(diff.abs().mean()) if diff.numel() else float("nan"),
        "max_abs_diff": float(diff.abs().max()) if diff.numel() else float("nan"),
        "n_elements": int(clean_vec.numel()),
    }


def summarize(rows, variants):
    summary = {
        "n_rows": len(rows),
        "variants": variants,
        "by_group": {},
        "by_head": {},
    }

    group_rows = [row for row in rows if row["unit_type"] == "group"]
    for group in sorted({row["group"] for row in group_rows}):
        subset = [row for row in group_rows if row["group"] == group]
        item = {
            "n": len(subset),
            "n_heads": int(subset[0]["n_heads"]) if subset else 0,
            "mean_clean_l2": mean([row["clean_l2"] for row in subset]),
            "mean_corrupt_l2": mean([row["corrupt_l2"] for row in subset]),
            "mean_diff_l2": mean([row["diff_l2"] for row in subset]),
            "mean_relative_diff_l2": mean([row["relative_diff_l2"] for row in subset]),
            "mean_cosine_similarity": mean([row["cosine_similarity"] for row in subset]),
            "by_variant": {},
        }
        for variant in variants:
            variant_subset = [row for row in subset if row["variant"] == variant]
            item["by_variant"][variant] = {
                "n": len(variant_subset),
                "mean_diff_l2": mean([row["diff_l2"] for row in variant_subset]),
                "mean_relative_diff_l2": mean([row["relative_diff_l2"] for row in variant_subset]),
                "mean_cosine_similarity": mean([row["cosine_similarity"] for row in variant_subset]),
            }
        summary["by_group"][group] = item

    head_rows = [row for row in rows if row["unit_type"] == "head"]
    for key in sorted({(row["layer"], row["head"]) for row in head_rows}):
        subset = [row for row in head_rows if (row["layer"], row["head"]) == key]
        summary["by_head"][f"L{key[0]}H{key[1]}"] = {
            "layer": int(key[0]),
            "head": int(key[1]),
            "n": len(subset),
            "mean_diff_l2": mean([row["diff_l2"] for row in subset]),
            "mean_relative_diff_l2": mean([row["relative_diff_l2"] for row in subset]),
            "mean_cosine_similarity": mean([row["cosine_similarity"] for row in subset]),
        }

    return summary


def main() -> None:
    args = parse_args()

    from tqdm.auto import tqdm

    from rha.config import load_runtime_config, resolve_repo_path
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer
    from rha.seed import set_seed
    from scripts.run_semantic_component_patching import (
        build_paired_examples,
        capture_query_cache,
    )

    selected_groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    groups = read_groups(resolve_repo_path(args.groups_csv), selected_groups)

    set_seed(args.seed)

    cfg = load_runtime_config(args.config)
    model, tokenizer = load_model_and_tokenizer(cfg)
    layout = inspect_model_layout(model)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    examples = build_paired_examples(
        tokenizer=tokenizer,
        variants=variants,
        n_per_variant=args.n_per_variant,
        target_tokens=args.target_tokens,
        needle_frac=args.needle_frac,
        seed_base=args.seed,
        max_secret_tries=args.max_secret_tries,
    )

    rows = []
    seen_heads = []
    for pairs in groups.values():
        for pair in pairs:
            if pair not in seen_heads:
                seen_heads.append(pair)

    for ex in tqdm(examples, desc="semantic activation delta"):
        clean_cache = capture_query_cache(
            model=model,
            tokenizer=tokenizer,
            layout=layout,
            prompt=ex["clean_prompt"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        corrupt_cache = capture_query_cache(
            model=model,
            tokenizer=tokenizer,
            layout=layout,
            prompt=ex["corrupt_prompt"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )

        for group, pairs in groups.items():
            row = {
                "unit_type": "group",
                "group": group,
                "layer": "",
                "head": "",
                "variant": ex["variant"],
                "seed": ex["seed"],
                "target_tokens": ex["target_tokens"],
                "prompt_tokens": ex["prompt_tokens"],
                "needle_frac": ex["needle_frac"],
                "clean_gold": ex["clean_gold"],
                "corrupt_gold": ex["corrupt_gold"],
                "n_heads": len(pairs),
            }
            row.update(metrics_for_vectors(
                vector_for_pairs(layout, clean_cache, pairs),
                vector_for_pairs(layout, corrupt_cache, pairs),
            ))
            rows.append(row)

        if args.include_head_rows:
            for layer, head in seen_heads:
                row = {
                    "unit_type": "head",
                    "group": "",
                    "layer": layer,
                    "head": head,
                    "variant": ex["variant"],
                    "seed": ex["seed"],
                    "target_tokens": ex["target_tokens"],
                    "prompt_tokens": ex["prompt_tokens"],
                    "needle_frac": ex["needle_frac"],
                    "clean_gold": ex["clean_gold"],
                    "corrupt_gold": ex["corrupt_gold"],
                    "n_heads": 1,
                }
                row.update(metrics_for_vectors(
                    vector_for_pairs(layout, clean_cache, [(layer, head)]),
                    vector_for_pairs(layout, corrupt_cache, [(layer, head)]),
                ))
                rows.append(row)

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, variants)
    summary.update(
        {
            "model_id": cfg.model_id,
            "n_examples": len(examples),
            "groups_csv": str(resolve_repo_path(args.groups_csv)),
            "target_tokens": args.target_tokens,
            "needle_frac": args.needle_frac,
            "include_head_rows": args.include_head_rows,
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
