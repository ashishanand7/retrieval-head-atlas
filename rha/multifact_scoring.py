"""Four-candidate adapter; intervention and decoding implementations stay frozen."""

from dataclasses import replace
import math

from rha.confirmation import answer_ids
from rha.multifact import LABELS
from rha.specificity_scoring import capture_reference, evaluate
from rha.temporal_scoring import prepare_prompt as prepare_three
from scripts.run_publication_temporal import assert_self_patch


def prepare_prompt(model, tokenizer, prompt, answers, max_len, chunk_size, max_new_tokens):
    if set(answers) != set(LABELS) or len(set(answers.values())) != 4:
        raise ValueError("Need all four distinct fact candidates")
    ids = {label: answer_ids(tokenizer, prompt, answer) for label, answer in answers.items()}
    if any(len(tokens) != 6 or any(tokenizer.decode([token]) != digit for token, digit in zip(tokens, answers[label]))
           for label, tokens in ids.items()):
        raise ValueError("Every candidate must be six single-digit tokens")
    prepared = prepare_three(model, tokenizer, prompt, {key: answers[key] for key in LABELS[:3]},
                             max_len, chunk_size, max_new_tokens)
    return replace(prepared, answers=dict(answers), candidate_ids=ids)


def assert_control(values, baseline):
    assert_self_patch(values, baseline)
    if values["generated_text"] != baseline["generated_text"]:
        raise RuntimeError("Control changed decoded text")
    if set(values["candidate_scores"]) != set(LABELS) or set(baseline["candidate_scores"]) != set(LABELS):
        raise RuntimeError("Control is missing a fact candidate")
    for label in LABELS:
        left, right = values["candidate_scores"][label], baseline["candidate_scores"][label]
        if (left["token_ids"] != right["token_ids"] or len(left["token_logprobs"]) != len(right["token_logprobs"])
                or any(not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-5
                       for a, b in zip(left["token_logprobs"], right["token_logprobs"]))):
            raise RuntimeError("Control changed a candidate token sequence or score")
