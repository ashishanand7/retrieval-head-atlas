"""Frozen prose tasks with matched value-swap and query-selection corruptions."""

import itertools
import random
import re
from collections import Counter

from rha.confirmation import answer_ids
from rha.specificity import SCOPES, boundary_metrics, digest, panel_groups, prior_ids, select_groups

FAMILIES = ("dispatch", "archive", "workshop", "reservation")
VIEWS = ("far_chat", "near_chat")
CORRUPTIONS = ("value_swap", "query_selection")
NAMES = ("Emma", "Jack", "Maya", "Nina", "Owen", "Paul", "Rose")
LABELS = ("clean", "corrupt", "decoy", "competitor_2")
FACTS = {
    "dispatch": (
        "The dispatch ledger lists a parcel for {name} under tracking code {code}. The receiving desk will use that entry when arranging delivery.",
        "A parcel addressed to {name} was entered in the shipping register. Its tracking code is {code}, and the entry is ready for the courier."),
    "archive": (
        "The archive accepted a collection deposited by {name}. Its catalogue code is {code}, which staff use when locating the collection.",
        "For the collection donated by {name}, the archivist recorded catalogue code {code}. The materials are available through the reading room."),
    "workshop": (
        "The workshop accepted an instrument from {name} for inspection. The service record gives its job code as {code}, and the instrument is awaiting review.",
        "An instrument belonging to {name} has a service file in the workshop. The file is identified by job code {code} and accompanies the inspection request."),
    "reservation": (
        "The booking office reserved a meeting room for {name}. The confirmation code for this reservation is {code}, and the booking has been approved.",
        "A meeting-room reservation was approved for {name}. The booking confirmation carries code {code}, which the office uses to identify the reservation."),
}
QUESTIONS = {
    "dispatch": ("What is the parcel tracking code for {name}?", "Give the tracking code recorded for the parcel for {name}."),
    "archive": ("What is the catalogue code for the collection deposited by {name}?", "Which catalogue code identifies the collection donated by {name}?"),
    "workshop": ("What is the job code for the instrument brought in by {name}?", "Give the service job code recorded for the instrument belonging to {name}."),
    "reservation": ("What is the reservation confirmation code for {name}?", "Which confirmation code identifies the meeting-room booking for {name}?"),
}
OPENINGS = (
    "The facilities team reviewed the arrangements for the shared entrance.",
    "Staff met to discuss how supplies should be stored during routine maintenance.",
    "The reception desk circulated an update about the visitor waiting area.",
    "The planning group examined the layout of the communal workspace.",
    "A caretaker checked the cupboards and the equipment used for cleaning.",
    "The administrative office collected comments on the internal notice board.",
    "The building coordinator reviewed the procedure for reporting minor repairs.",
    "A staff meeting considered the placement of signs along the main corridor.",
    "The maintenance team discussed access to the shared storage room.",
    "The office committee inspected the tables used for public consultations.",
    "An internal review considered how meeting areas are prepared between sessions.",
    "The reception team discussed the availability of seating near the entrance.",
)
MIDDLES = (
    "The discussion focused on keeping the area accessible throughout the working day.",
    "Members agreed that written instructions should remain easy to find and understand.",
    "The staff asked for clearer communication when temporary arrangements change.",
    "The review noted that routine checks should be recorded in the shared notes.",
    "Participants compared the existing arrangement with suggestions from regular users.",
    "The group agreed to avoid moving equipment while the area is in use.",
    "The coordinator requested a brief update after the next inspection.",
    "The discussion covered practical changes that would not interrupt normal activities.",
    "Staff agreed that concerns should be passed to the responsible team promptly.",
    "The review emphasized keeping passageways clear and shared materials organized.",
    "Members requested feedback from colleagues before changing the routine.",
    "The team noted that clear labels would help visitors navigate the building.",
)
CLOSINGS = (
    "No immediate change to the procedure was requested.",
    "The coordinator will include the comments in the next internal update.",
    "The existing arrangement remains in place while feedback is collected.",
    "The suggestions were retained for a later planning discussion.",
    "Staff will revisit the issue after the regular maintenance review.",
    "A short note was added to the shared workspace guidelines.",
    "The team will check whether the proposed arrangement is practical.",
    "The discussion ended with agreement to keep the area available for normal use.",
)


def reserved_ids(protocol, stage, references=False):
    settings = protocol["stages"][stage]
    offset = protocol["reference_seed_offset"] if references else 0
    return {(family, settings["seed_base"] + i * 100000 + j + offset)
            for i, family in enumerate(FAMILIES) for j in range(settings["n_per_family"])}


def identity_spec(protocol, stage, family, seed, references=False):
    settings = protocol["stages"][stage]
    offset = protocol["reference_seed_offset"] if references else 0
    j = seed - settings["seed_base"] - FAMILIES.index(family) * 100000 - offset
    if not 0 <= j < settings["n_per_family"]:
        raise ValueError("Identity is outside the reserved schedule")
    ordinal = FAMILIES.index(family) * settings["n_per_family"] + j
    target = ordinal % 4
    alternate = (target + 1 + (ordinal // 4) % 3) % 4
    rng = random.Random(seed)
    names = rng.sample(NAMES, 4)
    codes = [str(first) + f"{rng.randrange(100000):05d}" for first in rng.sample(range(1, 10), 4)]
    return {"variant": family, "seed": seed, "wording": settings["wording"], "target_slot": target,
            "alternate_slot": alternate, "names": names, "codes": codes}


def prose(seed):
    paragraphs = [" ".join(parts) for parts in itertools.product(OPENINGS, MIDDLES, CLOSINGS)]
    random.Random(seed + 901).shuffle(paragraphs)
    return paragraphs


def body_parts(spec, paragraphs, view, corruption=None):
    if view not in VIEWS or corruption not in (None, *CORRUPTIONS):
        raise ValueError("Unknown multi-fact view or corruption")
    codes = list(spec["codes"])
    a, b = spec["target_slot"], spec["alternate_slot"]
    if corruption == "value_swap":
        codes[a], codes[b] = codes[b], codes[a]
    facts = [FACTS[spec["variant"]][spec["wording"]].format(name=name, code=code)
             for name, code in zip(spec["names"], codes)]
    background = prose(spec["seed"])
    if not 0 <= paragraphs <= len(background):
        raise ValueError("Prose budget exceeds the fixed paragraph pool")
    before = round(paragraphs * .1) if view == "far_chat" else paragraphs
    blocks = background[:before] + facts + background[before:paragraphs]
    document = "\n\n".join(blocks)
    query = spec["names"][b if corruption == "query_selection" else a]
    question = QUESTIONS[spec["variant"]][spec["wording"]].format(name=query)
    body = ("Read the office notes and use the recorded facts to answer the question.\n\n"
            + document + "\n\nQuestion: " + question + "\nRespond with only the six-digit code.")
    return body, document, question


def render(tokenizer, body):
    return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


def calibrate(tokenizer, spec, target_tokens):
    def length(n):
        return len(tokenizer(render(tokenizer, body_parts(spec, n, "far_chat")[0]), add_special_tokens=False).input_ids)
    if length(0) >= target_tokens:
        raise ValueError("Target context is too short for four facts")
    low, high = 0, 16
    while length(high) < target_tokens:
        high *= 2
        if high > len(prose(spec["seed"])):
            raise ValueError("Target context exceeds available prose")
    if length(high) == target_tokens:
        return high
    while low + 1 < high:
        middle = (low + high) // 2
        if length(middle) <= target_tokens:
            low = middle
        else:
            high = middle
    if not low:
        raise ValueError("Need background paragraphs for the distance comparison")
    return low


def token_metadata(tokenizer, prompt, codes, max_len):
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if not 2 <= len(ids) <= max_len:
        raise ValueError("Multi-fact prompt would require truncation")
    def offset(character):
        prefix = tokenizer(prompt[:character], add_special_tokens=False).input_ids
        if ids[:len(prefix)] != prefix:
            raise ValueError("Unstable source/query token boundary")
        return len(prefix)
    question = offset(prompt.index("Question:"))
    sources = {}
    for code in codes:
        if prompt.count(code) != 1:
            raise ValueError("Every fact code must occur exactly once")
        start = offset(prompt.index(code))
        tokens = tokenizer(code, add_special_tokens=False).input_ids
        if len(tokens) != 6 or ids[start:start + 6] != tokens or start + 6 >= question:
            raise ValueError("Invalid fact-code token span")
        sources[code] = {"start": start, "end": start + 6, "distance_to_question": question - start - 6}
    return {"prompt_tokens": len(ids), "final_prompt_token_id": ids[-1], "question_token_start": question,
            "source_spans": sources}


def candidate_answers(spec):
    slots = [spec["target_slot"], spec["alternate_slot"]]
    slots += [i for i in range(4) if i not in slots]
    return dict(zip(LABELS, (spec["codes"][i] for i in slots)))


def build_examples(tokenizer, protocol, stage, forbidden_ids):
    schedules = [reserved_ids(protocol, name, ref) for name in ("smoke", "confirmation") for ref in (False, True)]
    if (any(not 1 <= s["n_per_family"] <= 1000 or s["wording"] not in (0, 1) for s in protocol["stages"].values())
            or protocol["stages"]["smoke"]["wording"] == protocol["stages"]["confirmation"]["wording"]
            or not 1 <= protocol["reference_seed_offset"] < 100000
            or sum(map(len, schedules)) != len({seed for schedule in schedules for _, seed in schedule})):
        raise ValueError("Invalid or overlapping study/reference/wording schedule")
    if {seed for _, seed in reserved_ids(protocol, stage) | reserved_ids(protocol, stage, True)} & {seed for _, seed in forbidden_ids}:
        raise ValueError("Study or reference seed overlaps prior data")
    examples, references = [], []
    for family, seed in sorted(reserved_ids(protocol, stage), key=lambda x: (FAMILIES.index(x[0]), x[1])):
        spec = identity_spec(protocol, stage, family, seed)
        reference = identity_spec(protocol, stage, family, seed + protocol["reference_seed_offset"], True)
        if set(spec["codes"]) & set(reference["codes"]):
            raise ValueError("Study/reference answer collision; revise before running")
        paragraphs = calibrate(tokenizer, spec, protocol["target_tokens"])
        ref_paragraphs = calibrate(tokenizer, reference, protocol["target_tokens"])
        identity = f"{family}:{seed}"
        answers = candidate_answers(spec)
        paired = {}
        for view in VIEWS:
            ref_prompt = render(tokenizer, body_parts(reference, ref_paragraphs, view)[0])
            ref = {"reference_id": f"{family}:{reference['seed']}/{view}", "paired_example_id": identity,
                   **reference, "view": view, "paragraphs": ref_paragraphs, "prompt": ref_prompt,
                   "prompt_sha256": digest(ref_prompt), "gold": reference["codes"][reference["target_slot"]],
                   **token_metadata(tokenizer, ref_prompt, reference["codes"], protocol["max_len"])}
            references.append(ref)
            clean = render(tokenizer, body_parts(spec, paragraphs, view)[0])
            clean_meta = token_metadata(tokenizer, clean, spec["codes"], protocol["max_len"])
            for mode in CORRUPTIONS:
                corrupt = render(tokenizer, body_parts(spec, paragraphs, view, mode)[0])
                corrupt_meta = token_metadata(tokenizer, corrupt, spec["codes"], protocol["max_len"])
                a = tokenizer(clean, add_special_tokens=False).input_ids
                b = tokenizer(corrupt, add_special_tokens=False).input_ids
                if len(a) != len(b) or a[-1] != b[-1]:
                    raise ValueError("Donor/recipient prompt positions do not align")
                ids = {label: answer_ids(tokenizer, clean, answer) for label, answer in answers.items()}
                for label, answer in answers.items():
                    if (len(ids[label]) != 6 or answer_ids(tokenizer, corrupt, answer) != ids[label]
                            or any(tokenizer.decode([token]) != digit for token, digit in zip(ids[label], answer))):
                        raise ValueError("Candidates must be stable six single-digit tokens")
                changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
                if not changed:
                    raise ValueError("Corruption changed no tokens")
                if mode == "query_selection" and min(changed) < clean_meta["question_token_start"]:
                    raise ValueError("Query-selection corruption changed the document")
                if mode == "value_swap":
                    permitted = {i for code in (answers["clean"], answers["corrupt"]) for i in
                                 range(clean_meta["source_spans"][code]["start"], clean_meta["source_spans"][code]["end"])}
                    if not set(changed) <= permitted:
                        raise ValueError("Value swap changed non-answer tokens")
                paired[view, mode] = a, b
                examples.append({"example_id": identity, "case_id": f"{identity}/{view}/{mode}", **spec,
                    "view": view, "corruption": mode, "paragraphs": paragraphs, "target_tokens": protocol["target_tokens"],
                    "clean_gold": answers["clean"], "corrupt_gold": answers["corrupt"],
                    "other_fact_codes": [answers["decoy"], answers["competitor_2"]],
                    "candidate_answers": answers, "candidate_token_ids": ids, "shared_prefix_tokens": 0,
                    "reference_id": ref["reference_id"] if mode == "query_selection" else None,
                    "resample_gold": ref["gold"] if mode == "query_selection" else None,
                    "clean_prompt": clean, "corrupt_prompt": corrupt,
                    "clean_prompt_sha256": digest(clean), "corrupt_prompt_sha256": digest(corrupt),
                    **clean_meta, "corrupt_source_spans": corrupt_meta["source_spans"],
                    "changed_token_positions": changed})
        for mode in CORRUPTIONS:
            for index in (0, 1):
                far, near = paired["far_chat", mode][index], paired["near_chat", mode][index]
                if len(far) != len(near) or Counter(far) != Counter(near):
                    raise ValueError("Far/near prompts must contain exactly the same tokens")
    return examples, references


def condition_specs(example, groups, protocol):
    selected = panel_groups(groups, protocol["distance_groups"])
    if example["corruption"] == "query_selection":
        selected.update(panel_groups(groups, protocol["ablation_groups"]))
    union = sorted({tuple(pair) for pairs in selected.values() for pair in pairs})
    specs = {}
    def add(name, action, scope, recipient, pairs=(), donor=None):
        specs[name] = {"action": action, "scope": scope, "recipient": recipient,
                       "pairs": list(map(tuple, pairs)), "donor": donor}
    for label in ("clean", "corrupt"):
        add(label, "baseline", "none", label)
        for scope in SCOPES:
            add(f"self_patch_{label}:{scope}", "patch", scope, label, union, label)
    add("positive_transfer", "residual", "prompt_and_decode", "corrupt", donor="clean")
    for name, pairs in panel_groups(groups, protocol["distance_groups"]).items():
        for scope in SCOPES:
            add(f"patch:{scope}:{name}", "patch", scope, "corrupt", pairs, "clean")
    if example["corruption"] == "query_selection":
        for name, pairs in panel_groups(groups, protocol["ablation_groups"]).items():
            add(f"zero:prompt_only:{name}", "zero", "prompt_only", "clean", pairs)
            add(f"resample:prompt_only:{name}", "resample", "prompt_only", "clean", pairs, "resample")
    return specs


def extra_metrics(text, scores, answers, resample_gold):
    numbers = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    other = {answers["decoy"], answers["competitor_2"]}
    margin = scores["clean"]["sum_logprob"] - max(score["sum_logprob"] for label, score in scores.items() if label != "clean")
    return {**boundary_metrics(text, answers["clean"], resample_gold),
            "clean_best_other_margin": margin, "clean_beats_all_candidates": int(margin > 0),
            "exact_other_fact": int(text.strip() in other),
            "first_number_is_clean": int(bool(numbers) and numbers[0] == answers["clean"]),
            "contains_clean_number": int(answers["clean"] in numbers),
            "contains_reference_number": int(resample_gold in numbers),
            "first_number_is_other_fact": int(bool(numbers) and numbers[0] in other)}
