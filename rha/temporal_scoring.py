"""Prefix-aligned donor patches with isolated caches for every continuation branch."""

import copy
from dataclasses import dataclass

import torch

from rha.confirmation import answer_ids
from rha.confirmation_scoring import checked_intervention
from rha.generation import cache_seq_len, one_step_logits, prefill_build_kv
from rha.interventions import OProjInputCacher
from rha.modeling import use_attn_impl
from rha.temporal import SCOPES, output_metrics


@dataclass
class PreparedPrompt:
    past: object
    last_token: torch.Tensor
    length: int
    answers: dict
    candidate_ids: dict


@torch.no_grad()
def prepare_prompt(model, tokenizer, prompt, answers, max_len, chunk_size, max_new_tokens):
    length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    if length < 2 or length > max_len or chunk_size < 1 or max_new_tokens < 1:
        raise ValueError("Invalid budget or temporal prompt would be truncated")
    if set(answers) != {"clean", "corrupt", "decoy"} or len(set(answers.values())) != 3:
        raise ValueError("Need three distinct answer candidates")
    ids = {label: answer_ids(tokenizer, prompt, answer) for label, answer in answers.items()}
    if len({len(value) for value in ids.values()}) != 1:
        raise ValueError("Answer candidates have different token lengths")
    if length + max(max_new_tokens, *(len(value) for value in ids.values())) > model.config.max_position_embeddings:
        raise ValueError("Prompt plus continuation exceeds model position limit")
    past, input_ids = prefill_build_kv(model, tokenizer, prompt, max_len, chunk_size)
    if input_ids.shape[1] != length or cache_seq_len(past) != length - 1:
        raise RuntimeError("Unexpected temporal prefill length")
    return PreparedPrompt(past, input_ids[:, -1:].clone(), length, dict(answers), ids)


def patch_active(scope, step):
    if scope not in (*SCOPES, "none") or step < 0:
        raise ValueError("Invalid temporal intervention scope or step")
    return scope == "prompt_and_decode" or (scope == "prompt_only" and step == 0) or (scope == "decode_only" and step > 0)


@torch.no_grad()
def aligned_step(model, layout, token, recipient_past, donor_past, pairs, scope, step, prompt_length):
    expected_length = prompt_length - 1 + step
    if cache_seq_len(recipient_past) != expected_length:
        raise RuntimeError("Recipient cache is not at the expected position")
    active = patch_active(scope, step)
    cache, donor_before = None, None
    if donor_past is not None:
        donor_before = cache_seq_len(donor_past)
        if donor_before != expected_length:
            raise RuntimeError("Donor and recipient positions are misaligned")
        # The same token tensor is used in both contexts, including recipient-chosen tokens.
        with OProjInputCacher(layout) as captured:
            _, donor_past = one_step_logits(model, token, donor_past)
        cache = captured.cache
        if cache_seq_len(donor_past) != expected_length + 1:
            raise RuntimeError("Donor step did not advance exactly one position")
    if active:
        if cache is None or not pairs:
            raise ValueError("Patching requires a current, aligned donor and nonempty heads")
        with checked_intervention(layout, pairs, cache):
            logits, recipient_past = one_step_logits(model, token, recipient_past)
    else:
        logits, recipient_past = one_step_logits(model, token, recipient_past)
    if cache_seq_len(recipient_past) != expected_length + 1:
        raise RuntimeError("Recipient step did not advance exactly one position")
    trace = {
        "step": step, "recipient_input_token_id": int(token.item()),
        "donor_input_token_id": int(token.item()) if donor_before is not None else None,
        "recipient_cache_length_before": expected_length, "donor_cache_length_before": donor_before,
        "patch_applied": active,
        # checked_intervention fails unless each selected layer fires exactly once.
        "verified_hook_calls": {str(layer): int(active) for layer, _ in pairs},
    }
    return logits, recipient_past, donor_past, trace


@torch.no_grad()
def continue_branch(model, tokenizer, layout, start, pairs, scope, prompt_length,
                    max_new_tokens, forced_ids=None):
    logits, recipient_start, donor_start, query_trace = start
    recipient = copy.deepcopy(recipient_start)
    donor = copy.deepcopy(donor_start)
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    eos = set(configured if isinstance(configured, (list, tuple)) else [configured])
    budget = len(forced_ids) if forced_ids is not None else max_new_tokens
    tokens, logprobs, trace = [], [], [query_trace]
    stop = "teacher_forced" if forced_ids is not None else "token_budget"
    for index in range(budget):
        token_id = forced_ids[index] if forced_ids is not None else int(logits.argmax(-1).item())
        tokens.append(token_id)
        logprobs.append(float(logits.float().log_softmax(-1)[0, token_id].cpu()))
        if forced_ids is None and token_id in eos:
            stop = "eos"
            break
        if index + 1 < budget:
            token = torch.tensor([[token_id]], device=model.device)
            logits, recipient, donor, event = aligned_step(
                model, layout, token, recipient, donor, pairs, scope, index + 1, prompt_length,
            )
            trace.append(event)
    return {"token_ids": tokens, "token_logprobs": logprobs, "steps": trace, "stop": stop}


@torch.no_grad()
def evaluate(model, tokenizer, layout, recipient, max_new_tokens, donor=None, pairs=None, scope="none"):
    pairs = list(pairs or [])
    if scope not in (*SCOPES, "none") or (scope == "none") != (donor is None and not pairs):
        raise ValueError("Baseline uses scope none; interventions require both donor and heads")
    if scope != "none" and (donor is None or not pairs):
        raise ValueError("Intervention requires donor and heads")
    if max_new_tokens < 1 or recipient.length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Generation budget exceeds model position limit")
    if donor is not None and (donor.length != recipient.length or not torch.equal(donor.last_token, recipient.last_token)
                              or donor.candidate_ids != recipient.candidate_ids or donor.answers != recipient.answers):
        raise ValueError("Donor/recipient prompt or candidate alignment mismatch")
    expected_before = recipient.length - 1
    if cache_seq_len(recipient.past) != expected_before or (donor is not None and cache_seq_len(donor.past) != expected_before):
        raise RuntimeError("Prepared prompt cache was mutated")
    with use_attn_impl(model, "sdpa"):
        start = aligned_step(model, layout, recipient.last_token, copy.deepcopy(recipient.past),
                             copy.deepcopy(donor.past) if donor is not None else None,
                             pairs, scope, 0, recipient.length)
        scores = {}
        for label, ids in recipient.candidate_ids.items():
            score = continue_branch(model, tokenizer, layout, start, pairs, scope, recipient.length, max_new_tokens, ids)
            score["sum_logprob"] = sum(score["token_logprobs"])
            score["mean_logprob"] = score["sum_logprob"] / len(ids)
            scores[label] = score
        generated = continue_branch(model, tokenizer, layout, start, pairs, scope, recipient.length, max_new_tokens)
    if cache_seq_len(start[1]) != recipient.length or (start[2] is not None and cache_seq_len(start[2]) != recipient.length):
        raise RuntimeError("A continuation branch mutated its shared initial cache")
    if cache_seq_len(recipient.past) != expected_before or (donor is not None and cache_seq_len(donor.past) != expected_before):
        raise RuntimeError("Evaluation mutated a prepared prompt cache")
    text = tokenizer.decode(generated["token_ids"], skip_special_tokens=True)
    return {
        "candidate_scores": scores,
        "clean_mean_logprob": scores["clean"]["mean_logprob"],
        "corrupt_mean_logprob": scores["corrupt"]["mean_logprob"],
        "decoy_mean_logprob": scores["decoy"]["mean_logprob"],
        "answer_logprob_margin": scores["clean"]["sum_logprob"] - scores["corrupt"]["sum_logprob"],
        "clean_decoy_logprob_margin": scores["clean"]["sum_logprob"] - scores["decoy"]["sum_logprob"],
        "first_clean_logprob": scores["clean"]["token_logprobs"][0],
        "first_answer_logprob_margin": scores["clean"]["token_logprobs"][0] - scores["corrupt"]["token_logprobs"][0],
        "generated_token_ids": generated["token_ids"], "generation_stop": generated["stop"],
        "generation_steps": generated["steps"], "generated_token_logprobs": generated["token_logprobs"],
        **output_metrics(text, generated["token_ids"], recipient.answers, recipient.candidate_ids),
    }
