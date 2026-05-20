#!/usr/bin/env python
import argparse
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_groups(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    groups: Dict[str, List[Tuple[int, int]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"]
            groups.setdefault(group, [])
            pair = (int(row["layer"]), int(row["head"]))
            if pair not in groups[group]:
                groups[group].append(pair)
    if not groups:
        raise ValueError(f"No groups found in {path}.")
    return groups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether non-address groups are required to use answer-address patching."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--groups-csv", default="configs/semantic_functional_groups.csv")
    parser.add_argument("--patch-group", default="answer_address")
    parser.add_argument(
        "--ablate-groups",
        default="query_tail,first_token_sink,non_address_core,query_tail_inactive_control,answer_address_inactive_control",
    )
    parser.add_argument("--out", default="artifacts_phase2/semantic_patch_interaction.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_patch_interaction_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-secret-tries", type=int, default=2000)
    return parser.parse_args()


class CompositeIntervention:
    def __init__(self, contexts):
        self.contexts = contexts
        self.stack = None

    def __enter__(self):
        self.stack = ExitStack()
        for ctx in self.contexts:
            self.stack.enter_context(ctx)
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)


def mean(values):
    import numpy as np

    return float(np.mean(values)) if values else float("nan")


def summarize(rows, variants, ablate_groups):
    summary = {
        "n_rows": len(rows),
        "variants": variants,
        "by_condition": {},
        "interactions": {},
    }

    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        item = {
            "n": len(subset),
            "mean_clean_logprob": mean([row["clean_logprob"] for row in subset]),
            "mean_corrupt_clean_logprob": mean([row["corrupt_clean_logprob"] for row in subset]),
            "mean_patched_logprob": mean([row["patched_logprob"] for row in subset]),
            "mean_delta_patch": mean([row["delta_patch"] for row in subset]),
            "mean_delta_vs_patch_only": mean([row["delta_vs_patch_only"] for row in subset]),
            "mean_recovery_fraction": mean([row["recovery_fraction"] for row in subset]),
            "positive_patch_examples": int(sum(1 for row in subset if row["delta_patch"] > 0)),
            "by_variant": {},
        }
        for variant in variants:
            variant_subset = [row for row in subset if row["variant"] == variant]
            item["by_variant"][variant] = {
                "n": len(variant_subset),
                "mean_delta_patch": mean([row["delta_patch"] for row in variant_subset]),
                "mean_delta_vs_patch_only": mean([row["delta_vs_patch_only"] for row in variant_subset]),
                "mean_recovery_fraction": mean([row["recovery_fraction"] for row in variant_subset]),
                "positive_patch_examples": int(sum(1 for row in variant_subset if row["delta_patch"] > 0)),
            }
        summary["by_condition"][condition] = item

    by_example = {}
    for row in rows:
        key = (row["variant"], row["seed"])
        by_example.setdefault(key, {})[row["condition"]] = row

    for group in ablate_groups:
        ablate_condition = f"ablate_only_{group}"
        patch_ablate_condition = f"patch_plus_ablate_{group}"
        patch_effect_under_ablation = []
        interaction_losses = []
        ablate_main_effects = []
        for condition_rows in by_example.values():
            if not all(
                name in condition_rows
                for name in ["patch_only", ablate_condition, patch_ablate_condition]
            ):
                continue
            patch_only = condition_rows["patch_only"]
            ablate_only = condition_rows[ablate_condition]
            patch_ablate = condition_rows[patch_ablate_condition]
            patch_effect = patch_only["delta_patch"]
            patched_when_ablated = patch_ablate["patched_logprob"] - ablate_only["patched_logprob"]
            patch_effect_under_ablation.append(patched_when_ablated)
            interaction_losses.append(patched_when_ablated - patch_effect)
            ablate_main_effects.append(ablate_only["delta_patch"])

        summary["interactions"][group] = {
            "n": len(patch_effect_under_ablation),
            "mean_ablate_main_effect": mean(ablate_main_effects),
            "mean_patch_effect_under_ablation": mean(patch_effect_under_ablation),
            "mean_interaction_loss": mean(interaction_losses),
            "patch_effect_under_ablation_positive": int(sum(1 for value in patch_effect_under_ablation if value > 0)),
            "interaction_loss_negative": int(sum(1 for value in interaction_losses if value < 0)),
        }

    return summary


def main() -> None:
    args = parse_args()

    from tqdm.auto import tqdm

    from rha.config import load_runtime_config, resolve_repo_path
    from rha.generation import mean_gold_logprob
    from rha.interventions import HeadAblator, HeadPatcher, spec_from_pairs
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer
    from rha.seed import set_seed
    from scripts.run_semantic_component_patching import (
        build_paired_examples,
        capture_query_cache,
    )

    groups = read_groups(resolve_repo_path(args.groups_csv))
    if args.patch_group not in groups:
        raise ValueError(f"Unknown patch group: {args.patch_group}")
    ablate_groups = [item.strip() for item in args.ablate_groups.split(",") if item.strip()]
    for group in ablate_groups:
        if group not in groups:
            raise ValueError(f"Unknown ablate group: {group}")

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

    patch_pairs = groups[args.patch_group]
    patch_spec = spec_from_pairs(patch_pairs)
    rows = []

    for ex in tqdm(examples, desc="semantic patch interaction"):
        clean_logprob = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["clean_prompt"],
            gold=ex["clean_gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        corrupt_clean_logprob = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["corrupt_prompt"],
            gold=ex["clean_gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        clean_cache = capture_query_cache(
            model=model,
            tokenizer=tokenizer,
            layout=layout,
            prompt=ex["clean_prompt"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        recovery_denominator = clean_logprob - corrupt_clean_logprob

        patch_only_lp = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["corrupt_prompt"],
            gold=ex["clean_gold"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
            intervention=HeadPatcher(layout, clean_cache, patch_spec),
            intervention_scope="query",
        )

        conditions = [("patch_only", patch_only_lp, args.patch_group, "")]
        for ablate_group in ablate_groups:
            ablate_spec = spec_from_pairs(groups[ablate_group])
            ablate_only_lp = mean_gold_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["corrupt_prompt"],
                gold=ex["clean_gold"],
                max_len=args.max_len,
                chunk_size=args.chunk_size,
                intervention=HeadAblator(layout, ablate_spec),
                intervention_scope="query",
            )
            conditions.append((f"ablate_only_{ablate_group}", ablate_only_lp, "", ablate_group))

            intervention = CompositeIntervention(
                [
                    HeadPatcher(layout, clean_cache, patch_spec),
                    HeadAblator(layout, ablate_spec),
                ]
            )
            lp = mean_gold_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["corrupt_prompt"],
                gold=ex["clean_gold"],
                max_len=args.max_len,
                chunk_size=args.chunk_size,
                intervention=intervention,
                intervention_scope="query",
            )
            conditions.append((f"patch_plus_ablate_{ablate_group}", lp, args.patch_group, ablate_group))

        for condition, patched_logprob, patch_group, ablate_group in conditions:
            delta_patch = patched_logprob - corrupt_clean_logprob
            recovery_fraction = (
                delta_patch / recovery_denominator
                if abs(recovery_denominator) > 1e-9
                else float("nan")
            )
            rows.append(
                {
                    "condition": condition,
                    "patch_group": patch_group,
                    "ablate_group": ablate_group,
                    "variant": ex["variant"],
                    "seed": ex["seed"],
                    "target_tokens": ex["target_tokens"],
                    "prompt_tokens": ex["prompt_tokens"],
                    "needle_frac": ex["needle_frac"],
                    "clean_gold": ex["clean_gold"],
                    "corrupt_gold": ex["corrupt_gold"],
                    "clean_logprob": clean_logprob,
                    "corrupt_clean_logprob": corrupt_clean_logprob,
                    "patched_logprob": patched_logprob,
                    "patch_only_logprob": patch_only_lp,
                    "delta_patch": delta_patch,
                    "delta_vs_patch_only": patched_logprob - patch_only_lp,
                    "recovery_denominator": recovery_denominator,
                    "recovery_fraction": recovery_fraction,
                }
            )

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, variants, ablate_groups)
    summary.update(
        {
            "model_id": cfg.model_id,
            "n_examples": len(examples),
            "groups_csv": str(resolve_repo_path(args.groups_csv)),
            "patch_group": args.patch_group,
            "ablate_groups": ablate_groups,
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
