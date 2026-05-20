#!/usr/bin/env python
import argparse
import csv
import json
import random
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
        description="Patch clean query-step component activations into corrupt semantic retrieval prompts."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--groups-csv", default="configs/semantic_functional_groups.csv")
    parser.add_argument("--groups", default="answer_address,non_address_core,strong13,core19,first_token_sink,query_tail")
    parser.add_argument("--out", default="artifacts_phase2/semantic_component_patching.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_component_patching_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-secret-tries", type=int, default=2000)
    return parser.parse_args()


def mean(values):
    import numpy as np

    return float(np.mean(values)) if values else float("nan")


def summarize(rows, variants, groups):
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
            "mean_clean_logprob": mean([row["clean_logprob"] for row in subset]),
            "mean_corrupt_clean_logprob": mean([row["corrupt_clean_logprob"] for row in subset]),
            "mean_patched_logprob": mean([row["patched_logprob"] for row in subset]),
            "mean_delta_patch": mean([row["delta_patch"] for row in subset]),
            "mean_recovery_fraction": mean([row["recovery_fraction"] for row in subset]),
            "positive_patch_examples": int(sum(1 for row in subset if row["delta_patch"] > 0)),
            "by_variant": {},
        }
        for variant in variants:
            variant_subset = [row for row in subset if row["variant"] == variant]
            item["by_variant"][variant] = {
                "n": len(variant_subset),
                "mean_delta_patch": mean([row["delta_patch"] for row in variant_subset]),
                "mean_recovery_fraction": mean([row["recovery_fraction"] for row in variant_subset]),
                "positive_patch_examples": int(sum(1 for row in variant_subset if row["delta_patch"] > 0)),
            }
        summary["by_group"][group] = item

    return summary


def sample_corrupt_secret(tokenizer, clean_prompt: str, clean_gold: str, seed: int, max_tries: int):
    from rha.prompts import seeded_secret

    clean_prompt_len = len(tokenizer(clean_prompt, add_special_tokens=False).input_ids)
    clean_gold_len = len(tokenizer(clean_gold, add_special_tokens=False).input_ids)
    rng = random.Random(seed + 90_000_001)

    for _ in range(max_tries):
        candidate = f"{rng.randint(100000, 999999)}"
        if candidate == clean_gold:
            continue
        if len(tokenizer(candidate, add_special_tokens=False).input_ids) != clean_gold_len:
            continue
        return candidate, clean_prompt_len

    # In Qwen digit strings are usually tokenized digit-by-digit, so this is a fallback guard.
    candidate = seeded_secret(seed + 90_000_001)
    if candidate == clean_gold:
        candidate = seeded_secret(seed + 90_000_002)
    return candidate, clean_prompt_len


def build_paired_examples(
    tokenizer,
    variants: List[str],
    n_per_variant: int,
    target_tokens: int,
    needle_frac: float,
    seed_base: int,
    max_secret_tries: int,
) -> List[Dict]:
    from rha.prompts import (
        build_prompt_with_filler,
        calibrate_filler_words,
        prompt_token_len,
        seeded_decoy,
        seeded_secret,
    )

    rows = []
    for variant_idx, variant in enumerate(variants):
        for i in range(n_per_variant):
            seed = seed_base + variant_idx * 100_000 + i
            clean_gold = seeded_secret(seed)
            decoy = seeded_decoy(seed)
            n_words = calibrate_filler_words(
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                needle_frac=needle_frac,
                variant=variant,
                secret=clean_gold,
                decoy=decoy,
                seed=seed,
            )
            clean_prompt = build_prompt_with_filler(n_words, needle_frac, variant, clean_gold, decoy, seed)
            corrupt_gold, clean_prompt_len = sample_corrupt_secret(
                tokenizer=tokenizer,
                clean_prompt=clean_prompt,
                clean_gold=clean_gold,
                seed=seed,
                max_tries=max_secret_tries,
            )
            corrupt_prompt = build_prompt_with_filler(n_words, needle_frac, variant, corrupt_gold, decoy, seed)
            corrupt_prompt_len = prompt_token_len(tokenizer, corrupt_prompt)

            if corrupt_prompt_len != clean_prompt_len:
                # Retry with fresh corrupt secrets until prompt lengths match.
                rng = random.Random(seed + 91_000_001)
                for _ in range(max_secret_tries):
                    candidate = f"{rng.randint(100000, 999999)}"
                    if candidate == clean_gold:
                        continue
                    candidate_prompt = build_prompt_with_filler(n_words, needle_frac, variant, candidate, decoy, seed)
                    if prompt_token_len(tokenizer, candidate_prompt) == clean_prompt_len:
                        corrupt_gold = candidate
                        corrupt_prompt = candidate_prompt
                        corrupt_prompt_len = clean_prompt_len
                        break

            if corrupt_prompt_len != clean_prompt_len:
                raise ValueError(
                    f"Could not build token-length-matched pair for variant={variant} seed={seed}: "
                    f"clean={clean_prompt_len}, corrupt={corrupt_prompt_len}"
                )

            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "target_tokens": target_tokens,
                    "needle_frac": needle_frac,
                    "clean_gold": clean_gold,
                    "corrupt_gold": corrupt_gold,
                    "clean_prompt": clean_prompt,
                    "corrupt_prompt": corrupt_prompt,
                    "prompt_tokens": clean_prompt_len,
                }
            )
    return rows


def capture_query_cache(model, tokenizer, layout, prompt: str, max_len: int, chunk_size: int):
    import torch

    from rha.generation import one_step_logits, prefill_build_kv
    from rha.interventions import OProjInputCacher
    from rha.modeling import use_attn_impl

    past, input_ids = prefill_build_kv(model, tokenizer, prompt, max_len=max_len, chunk_size=chunk_size)
    last_tok = input_ids[:, -1:]
    with torch.no_grad(), use_attn_impl(model, "sdpa"), OProjInputCacher(layout) as cacher:
        _logits, _past = one_step_logits(model, last_tok, past)
    return cacher.cache


def main() -> None:
    args = parse_args()

    from tqdm.auto import tqdm

    from rha.config import load_runtime_config, resolve_repo_path
    from rha.generation import mean_gold_logprob
    from rha.interventions import HeadPatcher, spec_from_pairs
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer
    from rha.seed import set_seed

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
    for ex in tqdm(examples, desc="semantic component patching"):
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
        corrupt_gold_logprob = mean_gold_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["corrupt_prompt"],
            gold=ex["corrupt_gold"],
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

        for group, pairs in groups.items():
            patched_logprob = mean_gold_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["corrupt_prompt"],
                gold=ex["clean_gold"],
                max_len=args.max_len,
                chunk_size=args.chunk_size,
                intervention=HeadPatcher(layout, clean_cache, spec_from_pairs(pairs)),
                intervention_scope="query",
            )
            delta_patch = patched_logprob - corrupt_clean_logprob
            recovery_fraction = (
                delta_patch / recovery_denominator
                if abs(recovery_denominator) > 1e-9
                else float("nan")
            )
            rows.append(
                {
                    "group": group,
                    "variant": ex["variant"],
                    "seed": ex["seed"],
                    "target_tokens": ex["target_tokens"],
                    "prompt_tokens": ex["prompt_tokens"],
                    "needle_frac": ex["needle_frac"],
                    "clean_gold": ex["clean_gold"],
                    "corrupt_gold": ex["corrupt_gold"],
                    "n_heads": len(pairs),
                    "clean_logprob": clean_logprob,
                    "corrupt_clean_logprob": corrupt_clean_logprob,
                    "corrupt_gold_logprob": corrupt_gold_logprob,
                    "patched_logprob": patched_logprob,
                    "delta_patch": delta_patch,
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

    summary = summarize(rows, variants, groups)
    summary.update(
        {
            "model_id": cfg.model_id,
            "n_examples": len(examples),
            "groups_csv": str(resolve_repo_path(args.groups_csv)),
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
