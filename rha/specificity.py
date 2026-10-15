"""Fixed paired task/boundary design and task-matched resampling references."""

import hashlib
import json
import random
import re
from collections import Counter

from rha.confirmation import VARIANTS, answer_ids, validate_pair
from rha.prompts import (NEEDLE_END, NEEDLE_START, _needle_sentence, calibrate_filler_words,
                         filler_words, seeded_decoy, seeded_secret)
from rha.temporal import CORRUPTIONS, prior_identities

SCOPES = ("prompt_only", "prompt_and_decode")
DISTANCE_VIEWS = ("far_chat", "near_chat", "short_chat")
BOUNDARY_VIEWS = ("far_chat", "far_raw", "far_raw_space")
ABLATION_VIEWS = DISTANCE_VIEWS + ("far_raw", "far_raw_space")
VIEW_CORRUPTIONS = tuple((view, mode) for view in DISTANCE_VIEWS for mode in CORRUPTIONS) + (
    ("far_raw", "different_first"), ("far_raw_space", "different_first"))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def boundary_metrics(text, gold, resample_gold):
    numbers = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    return {"verbatim_clean": int(text == gold),
            "generated_leading_whitespace": int(bool(text) and text[0].isspace()),
            "exact_resample": int(resample_gold is not None and text.strip() == resample_gold),
            "resample_number_content": int(resample_gold is not None and bool(numbers) and set(numbers) == {resample_gold})}


def select_groups(protocol, original):
    from scripts.run_publication_confirmation import make_groups
    if any(protocol[k] != original[k] for k in ("model_id", "num_layers", "num_heads")):
        raise ValueError("Specificity model and original head configuration differ")
    if (protocol["distance_groups"] != ["address_L22H7", "answer_address"]
            or protocol["ablation_groups"] != ["answer_address", "non_address_core"]
            or tuple(protocol["scopes"]) != SCOPES):
        raise ValueError("Unexpected specificity groups or scopes")
    names = set(protocol["distance_groups"] + protocol["ablation_groups"])
    return {name: pairs for name, pairs in make_groups(original, protocol["random_draws"]).items()
            if name in names or any(name.startswith(f"random_{group}_") for group in names)}


def panel_groups(groups, focal):
    return {name: pairs for name, pairs in groups.items()
            if name in focal or any(name.startswith(f"random_{group}_") for group in focal)}


def condition_specs(example, groups, protocol):
    distance = example["view"] in DISTANCE_VIEWS
    ablation = example["view"] in ABLATION_VIEWS and example["corruption"] == "different_first"
    selected = {}
    if distance:
        selected.update(panel_groups(groups, protocol["distance_groups"]))
    if ablation:
        selected.update(panel_groups(groups, protocol["ablation_groups"]))
    union = sorted({tuple(pair) for pairs in selected.values() for pair in pairs})
    specs = {}

    def add(name, action, scope, recipient, pairs=(), donor=None):
        specs[name] = {"action": action, "scope": scope, "recipient": recipient,
                       "pairs": list(map(tuple, pairs)), "donor": donor}

    for label in ("clean", "corrupt"):
        add(label, "baseline", "none", label)
        for scope in SCOPES if distance else ("prompt_only",):
            add(f"self_patch_{label}:{scope}", "patch", scope, label, union, label)
    add("positive_transfer", "residual", "prompt_and_decode", "corrupt", donor="clean")
    if distance:
        for name, pairs in panel_groups(groups, protocol["distance_groups"]).items():
            for scope in SCOPES:
                add(f"patch:{scope}:{name}", "patch", scope, "corrupt", pairs, "clean")
    if ablation:
        for name, pairs in panel_groups(groups, protocol["ablation_groups"]).items():
            add(f"zero:prompt_only:{name}", "zero", "prompt_only", "clean", pairs)
            add(f"resample:prompt_only:{name}", "resample", "prompt_only", "clean", pairs, "resample")
    return specs


def prior_ids(root, current_protocol):
    result = prior_identities(root, current_protocol)
    for path in sorted((root / "artifacts_publication").glob("*/references.jsonl")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest.get("protocol", {}).get("protocol") != current_protocol:
            for line in path.read_text().splitlines():
                reference = json.loads(line)
                result.add((reference["variant"], reference["seed"]))
    return result


def reserved_ids(protocol, stage, references=False):
    settings = protocol["stages"][stage]
    offset = protocol["reference_seed_offset"] if references else 0
    return {(variant, settings["seed_base"] + i * 100000 + j + offset)
            for i, variant in enumerate(VARIANTS) for j in range(settings["n_per_variant"])}


def make_body(variant, seed, words, distance):
    gold, decoy = seeded_secret(seed), seeded_decoy(seed)
    needle, question = _needle_sentence(variant, gold, decoy, seed)
    fact = needle.removeprefix(NEEDLE_START + " ").removesuffix(" " + NEEDLE_END)
    before_n = int(round(words * .1))
    before, after = filler_words(before_n, seed + 1), filler_words(words - before_n, seed + 2)
    if distance == "short":
        blocks = [fact]
    elif distance == "far":
        blocks = [before, fact, after]
    elif distance == "near":
        blocks = [before, after, fact]
    else:
        raise ValueError("Unknown source distance")
    # Uniform block boundaries preserve tokens when the same blocks are reordered.
    body = ("You are reading a long document. Answer the final question using only the relevant fact from the document.\n\n"
            "Document:\n\n" + "\n\n".join(blocks) + "\n\n"
            f"Distractor note: the number {decoy} is unrelated unless the document explicitly says otherwise.\n\n"
            f"Question: {question}\nAnswer with only the six digits.")
    return body


def render(tokenizer, body, view):
    if view.endswith("_chat"):
        return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    return body + (" " if view == "far_raw_space" else "")


def position_metadata(tokenizer, prompt, gold, max_len):
    tokens = tokenizer(prompt, add_special_tokens=False).input_ids
    if not 2 <= len(tokens) <= max_len:
        raise ValueError("Specificity prompt would require truncation")
    source_prefix = tokenizer(prompt[:prompt.index(gold)], add_special_tokens=False).input_ids
    question_prefix = tokenizer(prompt[:prompt.index("Question:")], add_special_tokens=False).input_ids
    start, question = len(source_prefix), len(question_prefix)
    ids = tokenizer(gold, add_special_tokens=False).input_ids
    if (tokens[:start] != source_prefix or tokens[:question] != question_prefix
            or tokens[start:start + len(ids)] != ids or question <= start + len(ids)):
        raise ValueError("Source/query token boundaries are not stable")
    return {"prompt_tokens": len(tokens), "final_prompt_token_id": tokens[-1],
            "source_token_start": start, "source_token_end": start + len(ids), "question_token_start": question,
            "source_to_question_tokens": question - start - len(ids),
            "source_to_answer_tokens": len(tokens) - start - len(ids)}


def build_examples(tokenizer, protocol, stage, forbidden_ids):
    if (protocol["needle_frac"] != .1 or protocol["target_tokens"] < 128
            or not 1 <= protocol["reference_seed_offset"] < 100000):
        raise ValueError("Invalid frozen specificity design")
    schedules = [reserved_ids(protocol, name, ref) for name in ("smoke", "confirmation") for ref in (False, True)]
    if any(not 1 <= setting["n_per_variant"] <= 1000 for setting in protocol["stages"].values()):
        raise ValueError("Invalid sample count")
    if sum(map(len, schedules)) != len({seed for schedule in schedules for _, seed in schedule}):
        raise ValueError("Specificity stage/reference identities overlap")
    if (reserved_ids(protocol, stage) | reserved_ids(protocol, stage, True)) & forbidden_ids:
        raise ValueError("Specificity study or reference identity overlaps prior data")
    examples, references = [], []
    for variant, seed in sorted(reserved_ids(protocol, stage), key=lambda pair: (VARIANTS.index(pair[0]), pair[1])):
        identity = f"{variant}:{seed}"
        gold, decoy = seeded_secret(seed), seeded_decoy(seed)
        ref_seed = seed + protocol["reference_seed_offset"]
        ref_gold, ref_decoy = seeded_secret(ref_seed), seeded_decoy(ref_seed)
        if len({gold, decoy, ref_gold, ref_decoy}) != 4:
            raise ValueError("Study/reference answers collide; revise protocol before running")
        words = calibrate_filler_words(tokenizer, protocol["target_tokens"], .1, variant, gold, decoy, seed)
        ref_words = calibrate_filler_words(tokenizer, protocol["target_tokens"], .1, variant, ref_gold, ref_decoy, ref_seed)
        clean = {view: render(tokenizer, make_body(variant, seed, words, view.split("_")[0]), view)
                 for view in set(DISTANCE_VIEWS + BOUNDARY_VIEWS)}
        positions = {view: position_metadata(tokenizer, prompt, gold, protocol["max_len"]) for view, prompt in clean.items()}
        if positions["far_chat"]["prompt_tokens"] != positions["near_chat"]["prompt_tokens"]:
            raise ValueError("Far/near context lengths differ; paired distance design requires exact matching")
        if Counter(tokenizer(clean["far_chat"], add_special_tokens=False).input_ids) != Counter(tokenizer(clean["near_chat"], add_special_tokens=False).input_ids):
            raise ValueError("Far/near views do not contain the same token multiset")
        if positions["far_chat"]["source_to_question_tokens"] <= positions["near_chat"]["source_to_question_tokens"]:
            raise ValueError("Near source is not closer to the query")
        refs = {}
        for view in ABLATION_VIEWS:
            prompt = render(tokenizer, make_body(variant, ref_seed, ref_words, view.split("_")[0]), view)
            refs[view] = {"reference_id": f"{variant}:{ref_seed}/{view}", "paired_example_id": identity,
                          "variant": variant, "seed": ref_seed, "view": view, "gold": ref_gold, "decoy": ref_decoy,
                          "filler_words": 0 if view == "short_chat" else ref_words,
                          "prompt": prompt, "prompt_sha256": digest(prompt),
                          **position_metadata(tokenizer, prompt, ref_gold, protocol["max_len"])}
            references.append(refs[view])
        for mode, shared in zip(CORRUPTIONS, (0, 1)):
            views = [view for view, corruption in VIEW_CORRUPTIONS if corruption == mode]
            rng = random.Random(seed + 190000001 + shared * 10000000)
            for _ in range(2000):
                other = str(rng.randint(100000, 999999))
                if other in (gold, decoy, ref_gold, ref_decoy) or other[:shared] != gold[:shared] or other[shared] == gold[shared]:
                    continue
                checks, candidates = {}, {}
                try:
                    for view in views:
                        corrupt = clean[view].replace(gold, other)
                        checks[view] = validate_pair(tokenizer, clean[view], corrupt, gold, other, protocol["max_len"])
                        ids = {label: answer_ids(tokenizer, clean[view], answer) for label, answer in
                               (("clean", gold), ("corrupt", other), ("decoy", decoy))}
                        for label, answer in (("clean", gold), ("corrupt", other), ("decoy", decoy)):
                            if (answer_ids(tokenizer, corrupt, answer) != ids[label] or len(ids[label]) != 6
                                    or any(tokenizer.decode([token]) != digit for token, digit in zip(ids[label], answer))):
                                raise ValueError("Candidates must be six stable single-digit tokens")
                        candidates[view] = ids
                except ValueError:
                    continue
                break
            else:
                raise ValueError("Could not construct aligned corruption across all views")
            for view in views:
                prompt, corrupt = clean[view], clean[view].replace(gold, other)
                reference = refs.get(view) if mode == "different_first" else None
                examples.append({"example_id": identity, "case_id": f"{identity}/{view}/{mode}",
                    "variant": variant, "seed": seed, "view": view, "corruption": mode, "shared_prefix_tokens": shared,
                    "marker_mode": "unmarked", "target_tokens": None if view == "short_chat" else protocol["target_tokens"],
                    "filler_words": 0 if view == "short_chat" else words, "clean_gold": gold, "corrupt_gold": other,
                    "unrelated_decoy": decoy, "reference_id": reference["reference_id"] if reference else None,
                    "resample_gold": ref_gold if reference else None,
                    "clean_prompt": prompt, "corrupt_prompt": corrupt,
                    "clean_prompt_sha256": digest(prompt), "corrupt_prompt_sha256": digest(corrupt),
                    "candidate_token_ids": candidates[view], **positions[view], **checks[view]})
    return examples, references
