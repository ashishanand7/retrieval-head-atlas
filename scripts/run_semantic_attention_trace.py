#!/usr/bin/env python
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEEDLE_START = "[NEEDLE_START]"
NEEDLE_END = "[NEEDLE_END]"


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
        description="Trace query-step attention mass from selected heads to gold, needle, and distractor spans."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--heads-csv", default="configs/semantic_core_heads.csv")
    parser.add_argument("--out", default="artifacts_phase2/semantic_attention_trace.csv")
    parser.add_argument("--summary-out", default="artifacts_phase2/semantic_attention_trace_summary.json")
    parser.add_argument("--variants", default="literal,alias,paraphrase,relational,distractor_heavy")
    parser.add_argument("--n-per-variant", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=19)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--needle-frac", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def char_span_to_token_span(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> Tuple[int, int]:
    tok_start = None
    tok_end = None
    for idx, (start, end) in enumerate(offsets):
        if end > char_start and start < char_end:
            if tok_start is None:
                tok_start = idx
            tok_end = idx + 1
    if tok_start is None or tok_end is None:
        return -1, -1
    return tok_start, tok_end


def span_mass(attn, span: Tuple[int, int]) -> float:
    start, end = span
    if start < 0 or end <= start or start >= len(attn):
        return 0.0
    end = min(end, len(attn))
    return float(attn[start:end].sum())


def span_max(attn, span: Tuple[int, int]) -> float:
    start, end = span
    if start < 0 or end <= start or start >= len(attn):
        return 0.0
    end = min(end, len(attn))
    return float(attn[start:end].max())


def span_rank(attn, span: Tuple[int, int]) -> int:
    best = span_max(attn, span)
    if best <= 0.0:
        return -1
    return int((attn > best).sum()) + 1


def mass_over_spans(attn, spans: Iterable[Tuple[int, int]]) -> float:
    return float(sum(span_mass(attn, span) for span in spans))


def find_token_spans(tokenizer, prompt: str, gold: str) -> Dict:
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])

    needle_start_char = prompt.rfind(NEEDLE_START)
    needle_end_marker = prompt.find(NEEDLE_END, needle_start_char + len(NEEDLE_START))
    if needle_start_char < 0 or needle_end_marker < 0:
        raise ValueError("Could not locate needle markers in prompt.")

    needle_end_char = needle_end_marker + len(NEEDLE_END)
    gold_char = prompt.find(str(gold), needle_start_char, needle_end_char)
    if gold_char < 0:
        raise ValueError("Could not locate gold answer inside needle markers.")

    six_digit_spans = []
    for match in re.finditer(r"\b\d{6}\b", prompt):
        if match.group(0) == str(gold) and match.start() == gold_char:
            continue
        six_digit_spans.append(char_span_to_token_span(offsets, match.start(), match.end()))

    return {
        "gold_span": char_span_to_token_span(offsets, gold_char, gold_char + len(str(gold))),
        "needle_span": char_span_to_token_span(offsets, needle_start_char, needle_end_char),
        "distractor_spans": [span for span in six_digit_spans if span[0] >= 0],
        "token_count": len(enc["input_ids"]),
    }


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
            "mean_gold_mass": mean([row["gold_mass"] for row in subset]),
            "mean_needle_mass": mean([row["needle_mass"] for row in subset]),
            "mean_distractor_mass": mean([row["distractor_mass"] for row in subset]),
            "mean_gold_rank": mean([row["gold_rank"] for row in subset if row["gold_rank"] > 0]),
            "argmax_in_gold_rate": mean([row["argmax_in_gold"] for row in subset]),
            "argmax_in_needle_rate": mean([row["argmax_in_needle"] for row in subset]),
            "by_variant": {},
        }
        for variant in variants:
            variant_subset = [row for row in subset if row["variant"] == variant]
            item["by_variant"][variant] = {
                "n": len(variant_subset),
                "mean_gold_mass": mean([row["gold_mass"] for row in variant_subset]),
                "mean_needle_mass": mean([row["needle_mass"] for row in variant_subset]),
                "mean_distractor_mass": mean([row["distractor_mass"] for row in variant_subset]),
                "argmax_in_needle_rate": mean([row["argmax_in_needle"] for row in variant_subset]),
            }
        summary["by_head"].append(item)

    summary["by_head"].sort(key=lambda row: row["mean_gold_mass"], reverse=True)

    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        summary["by_variant"][variant] = {
            "n": len(subset),
            "mean_gold_mass": mean([row["gold_mass"] for row in subset]),
            "mean_needle_mass": mean([row["needle_mass"] for row in subset]),
            "mean_distractor_mass": mean([row["distractor_mass"] for row in subset]),
        }

    return summary


def main() -> None:
    args = parse_args()

    import torch
    from tqdm.auto import tqdm

    from rha.config import load_runtime_config, resolve_repo_path
    from rha.generation import cache_seq_len, prefill_build_kv
    from rha.modeling import inspect_model_layout, load_model_and_tokenizer, use_attn_impl
    from rha.prompts import build_dataset
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
    for layer, head in pairs:
        if layer < 0 or layer >= len(layout.layers):
            raise ValueError(f"Layer {layer} is out of range.")
        if head < 0 or head >= layout.num_heads:
            raise ValueError(f"Head {head} is out of range.")

    rows = []
    for ex in tqdm(examples, desc="semantic attention trace"):
        spans = find_token_spans(tokenizer, ex["prompt"], ex["gold"])
        past, input_ids = prefill_build_kv(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            max_len=args.max_len,
            chunk_size=args.chunk_size,
        )
        last_tok = input_ids[:, -1:]
        past_len = cache_seq_len(past)
        attention_mask = torch.ones((last_tok.shape[0], past_len + 1), device=model.device, dtype=torch.long)
        position_ids = torch.tensor([[past_len]], device=model.device, dtype=torch.long)

        with torch.no_grad(), use_attn_impl(model, "eager"):
            out = model(
                input_ids=last_tok,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        for layer, head in pairs:
            attn = out.attentions[layer][0, head, -1, :].detach().float().cpu().numpy()
            gold_span = spans["gold_span"]
            needle_span = spans["needle_span"]
            argmax_idx = int(attn.argmax())
            rows.append(
                {
                    "variant": ex["variant"],
                    "seed": ex["seed"],
                    "target_tokens": ex["target_tokens"],
                    "actual_tokens": spans["token_count"],
                    "needle_frac": ex["needle_frac"],
                    "gold": ex["gold"],
                    "layer": layer,
                    "head": head,
                    "gold_span_start": gold_span[0],
                    "gold_span_end": gold_span[1],
                    "needle_span_start": needle_span[0],
                    "needle_span_end": needle_span[1],
                    "n_distractor_spans": len(spans["distractor_spans"]),
                    "gold_mass": span_mass(attn, gold_span),
                    "gold_max": span_max(attn, gold_span),
                    "gold_rank": span_rank(attn, gold_span),
                    "needle_mass": span_mass(attn, needle_span),
                    "distractor_mass": mass_over_spans(attn, spans["distractor_spans"]),
                    "argmax_token_index": argmax_idx,
                    "argmax_in_gold": int(gold_span[0] <= argmax_idx < gold_span[1]),
                    "argmax_in_needle": int(needle_span[0] <= argmax_idx < needle_span[1]),
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
