"""Frozen temporal-study groups, paired corruptions, and output measurements."""

import hashlib
import json
import random
import re

from rha.confirmation import VARIANTS, answer_ids, generation_metrics, historical_ids, render_prompt, validate_pair
from rha.prompts import build_prompt_with_filler, calibrate_filler_words, seeded_decoy, seeded_secret

SCOPES = ("prompt_only", "decode_only", "prompt_and_decode")
CORRUPTIONS = ("different_first", "same_first")


def select_groups(protocol, original):
    # Use all original group indices before filtering, so control seeds do not shift.
    from scripts.run_publication_confirmation import make_groups
    if any(protocol[field] != original[field] for field in ("model_id", "num_heads", "num_layers")):
        raise ValueError("Temporal and original head protocols disagree")
    groups = make_groups(original, protocol["random_draws"])
    focal = protocol["focal_groups"]
    if focal != ["address_L22H7", "answer_address"]:
        raise ValueError("This protocol freezes the two address groups")
    return {name: heads for name, heads in groups.items()
            if name in focal or any(name.startswith(f"random_{group}_") for group in focal)}


def condition_names(groups):
    return (["clean", "corrupt"]
            + [f"self_patch_{label}:{scope}" for label in ("clean", "corrupt") for scope in SCOPES]
            + [f"patch:{scope}:{name}" for name in groups for scope in SCOPES])


def prior_identities(root, current_protocol):
    identities = historical_ids(root / "artifacts_phase2")
    for path in sorted((root / "artifacts_publication").glob("*/examples.jsonl")):
        manifest_path = path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("protocol", {}).get("protocol") == current_protocol:
                continue  # Explicit repeat attempts retain their reserved temporal identities.
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    identities.add((row["variant"], row["seed"]))
    return identities


def build_examples(tokenizer, protocol, stage, forbidden_ids):
    settings = protocol["stages"][stage]
    n, seed_base = settings["n_per_variant"], settings["seed_base"]
    if not 1 <= n <= 1000 or protocol["target_tokens"] < 128 or not 0 <= protocol["needle_frac"] <= 1:
        raise ValueError("Invalid temporal sample count or context settings")
    stage_ids = [{base["seed_base"] + index * 100000 + offset for index in range(len(VARIANTS))
                  for offset in range(base["n_per_variant"])} for base in protocol["stages"].values()]
    if len(stage_ids) != 2 or stage_ids[0] & stage_ids[1]:
        raise ValueError("Smoke and confirmation seed schedules overlap")
    examples = []
    for index, variant in enumerate(VARIANTS):
        for offset in range(n):
            seed = seed_base + index * 100000 + offset
            if (variant, seed) in forbidden_ids:
                raise ValueError(f"Temporal identity overlaps prior data: {variant}/{seed}")
            gold, decoy = seeded_secret(seed), seeded_decoy(seed)
            if gold == decoy:
                raise ValueError("Clean answer and unrelated distractor collide; revise the protocol")
            words = calibrate_filler_words(tokenizer, protocol["target_tokens"], protocol["needle_frac"], variant, gold, decoy, seed)
            base = build_prompt_with_filler(words, protocol["needle_frac"], variant, gold, decoy, seed)
            clean = render_prompt(tokenizer, base, "unmarked", "chat")
            if not 2 <= len(tokenizer(clean, add_special_tokens=False).input_ids) <= protocol["max_len"]:
                raise ValueError("Clean temporal prompt would require truncation")
            for answer in (gold, decoy):
                ids = answer_ids(tokenizer, clean, answer)
                if len(ids) != 6 or any(tokenizer.decode([token]) != digit for token, digit in zip(ids, answer)):
                    raise ValueError("Temporal experiment requires one token per digit")
            for corruption, shared in zip(CORRUPTIONS, (0, 1)):
                rng = random.Random(seed + 190000001 + shared * 10000000)
                for _ in range(2000):
                    other = str(rng.randint(100000, 999999))
                    if other in (gold, decoy) or other[:shared] != gold[:shared] or other[shared] == gold[shared]:
                        continue
                    corrupt = clean.replace(gold, other)
                    try:
                        checks = validate_pair(tokenizer, clean, corrupt, gold, other, protocol["max_len"])
                        ids = {label: answer_ids(tokenizer, clean, answer)
                               for label, answer in (("clean", gold), ("corrupt", other), ("decoy", decoy))}
                        for label, answer in (("clean", gold), ("corrupt", other), ("decoy", decoy)):
                            if answer_ids(tokenizer, corrupt, answer) != ids[label]:
                                raise ValueError("Candidate tokenization changes across contexts")
                            if len(ids[label]) != 6 or any(tokenizer.decode([token]) != digit for token, digit in zip(ids[label], answer)):
                                raise ValueError("Temporal experiment requires one token per digit")
                        differing = [i for i, pair in enumerate(zip(ids["clean"], ids["corrupt"])) if pair[0] != pair[1]]
                        if differing[0] != shared:
                            raise ValueError("Token-level common prefix does not match the corruption stratum")
                    except ValueError:
                        continue
                    break
                else:
                    raise ValueError(f"No valid {corruption} corruption for {variant}/{seed}")
                examples.append({
                    "example_id": f"{variant}:{seed}", "case_id": f"{variant}:{seed}/{corruption}",
                    "variant": variant, "seed": seed, "corruption": corruption,
                    "marker_mode": "unmarked", "prompt_format": "chat", "shared_prefix_tokens": shared,
                    "target_tokens": protocol["target_tokens"], "needle_frac": protocol["needle_frac"],
                    "filler_words": words, "clean_gold": gold, "corrupt_gold": other, "unrelated_decoy": decoy,
                    "clean_prompt": clean, "corrupt_prompt": corrupt, "candidate_token_ids": ids,
                    "final_prompt_token_id": tokenizer(clean, add_special_tokens=False).input_ids[-1],
                    "clean_prompt_sha256": hashlib.sha256(clean.encode()).hexdigest(),
                    "corrupt_prompt_sha256": hashlib.sha256(corrupt.encode()).hexdigest(), **checks,
                })
    return examples


def output_metrics(text, generated, answers, candidate_ids):
    ids = candidate_ids["clean"]
    differing = next(i for i, pair in enumerate(zip(ids, candidate_ids["corrupt"])) if pair[0] != pair[1])
    numbers = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    metrics = generation_metrics(text, answers["clean"], answers["corrupt"])
    metrics.update({"exact_decoy": int(text.strip() == answers["decoy"]),
                    "sole_number_is_decoy": int(numbers == [answers["decoy"]]),
                    "clean_number_content": int(bool(numbers) and set(numbers) == {answers["clean"]})})
    for label in ("clean", "corrupt"):
        target = candidate_ids[label]
        suffix = [int(i < len(generated) and generated[i] == target[i]) for i in range(differing, len(target))]
        metrics[f"first_token_is_{label}"] = int(bool(generated) and generated[0] == target[0])
        metrics[f"first_differing_token_is_{label}"] = suffix[0]
        metrics[f"aligned_suffix_exact_{label}"] = int(all(suffix))
        metrics[f"aligned_suffix_fraction_{label}"] = sum(suffix) / len(suffix)
    return metrics
