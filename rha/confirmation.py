"""Prompt and provenance checks for the first publication confirmation batch."""

import csv
import hashlib
import random
import re
from pathlib import Path

from rha.prompts import (
    NEEDLE_END, NEEDLE_START, build_prompt_with_filler, calibrate_filler_words,
    seeded_decoy, seeded_secret,
)

VARIANTS = ("literal", "alias", "paraphrase", "relational", "distractor_heavy")


def historical_ids(directory):
    keys = set()
    paths = sorted(Path(directory).glob("semantic_*.csv"))
    if not paths:
        raise ValueError(f"No historical artifacts found in {directory}; cannot verify seed separation")
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if "variant" in row and "seed" in row:
                    keys.add((row["variant"], int(row["seed"])))
    if not keys:
        raise ValueError("Historical artifact files contain no example identities")
    return keys


def render_prompt(tokenizer, prompt, marker_mode, prompt_format):
    if marker_mode not in ("marked", "unmarked"):
        raise ValueError(f"Unknown marker mode: {marker_mode}")
    if prompt.count(NEEDLE_START) != 1 or prompt.count(NEEDLE_END) != 1:
        raise ValueError("Expected exactly one pair of needle markers")
    if marker_mode == "unmarked":
        prompt = prompt.replace(NEEDLE_START + " ", "").replace(" " + NEEDLE_END, "")
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
    if prompt_format != "raw":
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    return prompt


def answer_ids(tokenizer, prompt, answer):
    prefix = tokenizer(prompt, add_special_tokens=False).input_ids
    complete = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not ids or complete != prefix + ids:
        raise ValueError("Answer tokenization is not a stable continuation of this prompt")
    return ids


def validate_pair(tokenizer, clean, corrupt, gold, other, max_len):
    if gold == other or clean.count(gold) != 1 or corrupt.count(other) != 1:
        raise ValueError("Answers must be distinct and occur at exactly one source location")
    if clean.replace(gold, "<ANSWER>") != corrupt.replace(other, "<ANSWER>"):
        raise ValueError("Clean/corrupt prompts differ outside the answer span")
    a = tokenizer(clean, add_special_tokens=False).input_ids
    b = tokenizer(corrupt, add_special_tokens=False).input_ids
    if len(a) != len(b) or len(a) < 2 or len(a) > max_len:
        raise ValueError("Prompt lengths must match and fit without truncation")
    gold_ids = answer_ids(tokenizer, clean, gold)
    other_ids = answer_ids(tokenizer, corrupt, other)
    answer_ids(tokenizer, clean, other)
    answer_ids(tokenizer, corrupt, gold)
    if len(gold_ids) != len(other_ids) or a[-1] != b[-1]:
        raise ValueError("Answer token counts or final query tokens do not match")
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not differing:
        raise ValueError("Answer substitution changed no tokens")
    return {
        "prompt_tokens": len(a), "answer_tokens": len(gold_ids),
        "changed_token_first": differing[0], "changed_token_last": differing[-1],
        "first_answer_tokens_differ": gold_ids[0] != other_ids[0],
        "gold_char_start": clean.index(gold), "corrupt_gold_char_start": corrupt.index(other),
    }


def build_examples(tokenizer, n_per_variant, target_tokens, needle_frac, seed_base,
                   prompt_format, max_len, forbidden_ids):
    if n_per_variant < 1 or n_per_variant >= 100000 or target_tokens < 128:
        raise ValueError("Invalid sample count or target context length")
    if not 0 <= needle_frac <= 1:
        raise ValueError("needle_frac must lie in [0, 1]")
    examples = []
    for variant_index, variant in enumerate(VARIANTS):
        for i in range(n_per_variant):
            seed = seed_base + variant_index * 100000 + i
            if (variant, seed) in forbidden_ids:
                raise ValueError(f"Example overlaps historical discovery data: {variant}/{seed}")
            gold, decoy = seeded_secret(seed), seeded_decoy(seed)
            if gold == decoy:
                raise ValueError(f"Gold and distractor collide for seed {seed}; revise the protocol")
            words = calibrate_filler_words(tokenizer, target_tokens, needle_frac, variant, gold, decoy, seed)
            base = build_prompt_with_filler(words, needle_frac, variant, gold, decoy, seed)
            clean = {mode: render_prompt(tokenizer, base, mode, prompt_format) for mode in ("marked", "unmarked")}
            for prompt in clean.values():
                length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                if length < 2 or length > max_len:
                    raise ValueError(f"Clean prompt would require truncation for {variant}/{seed}")
                answer_ids(tokenizer, prompt, gold)
            rng = random.Random(seed + 90000001)
            for _ in range(2000):
                other = str(rng.randint(100000, 999999))
                if other in (gold, decoy):
                    continue
                corrupt_base = build_prompt_with_filler(words, needle_frac, variant, other, decoy, seed)
                corrupt = {mode: render_prompt(tokenizer, corrupt_base, mode, prompt_format) for mode in clean}
                try:
                    checks = {mode: validate_pair(tokenizer, clean[mode], corrupt[mode], gold, other, max_len) for mode in clean}
                except ValueError:
                    continue
                break
            else:
                raise ValueError(f"Cannot construct a valid, untruncated pair for {variant}/{seed}")
            for mode in clean:
                examples.append({
                    "example_id": f"{variant}:{seed}", "variant": variant, "seed": seed,
                    "marker_mode": mode, "prompt_format": prompt_format,
                    "target_tokens": target_tokens, "needle_frac": needle_frac,
                    "filler_words": words, "clean_gold": gold, "corrupt_gold": other,
                    "clean_prompt": clean[mode], "corrupt_prompt": corrupt[mode],
                    "clean_prompt_sha256": hashlib.sha256(clean[mode].encode()).hexdigest(),
                    "corrupt_prompt_sha256": hashlib.sha256(corrupt[mode].encode()).hexdigest(),
                    **checks[mode],
                })
    return examples


def generation_metrics(text, gold, other):
    stripped = text.strip()
    candidates = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    return {
        "generated_text": text,
        "exact_clean": int(stripped == gold), "exact_corrupt": int(stripped == other),
        "six_digits_only": int(re.fullmatch(r"\d{6}", stripped) is not None),
        "sole_number_is_clean": int(candidates == [gold]),
        "sole_number_is_corrupt": int(candidates == [other]),
    }
