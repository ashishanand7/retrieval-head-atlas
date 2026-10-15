"""Query ablation/resampling and a checked final-residual transfer calibration."""

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

from rha.confirmation_scoring import checked_intervention
from rha.generation import cache_seq_len, one_step_logits, prefill_build_kv
from rha.interventions import OProjInputCacher
from rha.modeling import use_attn_impl
from rha.temporal import output_metrics
from rha.temporal_scoring import evaluate as temporal_evaluate
from rha.temporal_scoring import prepare_prompt


@dataclass
class ResampledQuery:
    reference_id: str
    input_token_id: int
    position: int
    activations: dict


@torch.no_grad()
def capture_reference(model, tokenizer, layout, reference, max_len, chunk_size):
    length = len(tokenizer(reference["prompt"], add_special_tokens=False).input_ids)
    if length != reference["prompt_tokens"] or not 2 <= length <= min(max_len, model.config.max_position_embeddings):
        raise ValueError("Resampling reference would be truncated or has incorrect length")
    past, ids = prefill_build_kv(model, tokenizer, reference["prompt"], max_len, chunk_size)
    if cache_seq_len(past) != length - 1 or ids.shape[1] != length:
        raise RuntimeError("Invalid resampling prefill")
    with use_attn_impl(model, "sdpa"), OProjInputCacher(layout) as captured:
        one_step_logits(model, ids[:, -1:], past)
    if any(tuple(value.shape) != (1, 1, layout.hidden_size) for value in captured.cache.values()):
        raise RuntimeError("Resampling cache is not a single query position")
    return ResampledQuery(reference["reference_id"], int(ids[0, -1]), length - 1, captured.cache)


@contextmanager
def final_residual(model, replacement=None):
    if model.config.model_type != "qwen2" or not hasattr(model.model, "norm"):
        raise ValueError("Positive calibration requires the inspected Qwen2 final norm")
    state = {"calls": 0, "value": None}

    def hook(_module, inputs):
        value = inputs[0]
        if tuple(value.shape) != (1, 1, model.config.hidden_size):
            raise ValueError("Final residual must be a single current-position vector")
        state["calls"] += 1
        if replacement is None:
            state["value"] = value.detach().clone()
            return None
        if replacement.shape != value.shape or replacement.dtype != value.dtype or replacement.device != value.device:
            raise ValueError("Incompatible final-residual donor")
        return (replacement.clone(),) + tuple(inputs[1:])

    handle = model.model.norm.register_forward_pre_hook(hook)
    try:
        yield state
        if state["calls"] != 1:
            raise RuntimeError("Final residual hook did not fire exactly once")
    finally:
        handle.remove()


@torch.no_grad()
def intervention_step(model, layout, token, recipient_past, donor_past, action, pairs, step, prompt_length, resample=None):
    position = prompt_length - 1 + step
    if cache_seq_len(recipient_past) != position:
        raise RuntimeError("Incorrect recipient cache position")
    active = action == "residual" or step == 0
    donor_logits, source = None, None
    donor_before = None
    if action == "residual":
        donor_before = cache_seq_len(donor_past)
        if donor_past is None or donor_before != position:
            raise RuntimeError("Positive donor prefix position is misaligned")
        with final_residual(model) as captured:
            donor_logits, donor_past = one_step_logits(model, token, donor_past)
        source = captured["value"]
        if cache_seq_len(donor_past) != position + 1:
            raise RuntimeError("Positive donor did not advance one token")
    if action == "residual":
        context = final_residual(model, source)
    elif active:
        context = checked_intervention(layout, pairs, resample.activations if action == "resample" else None)
    else:
        context = nullcontext()
    with context:
        logits, recipient_past = one_step_logits(model, token, recipient_past)
    if cache_seq_len(recipient_past) != position + 1:
        raise RuntimeError("Recipient did not advance one token")
    error = None
    if action == "residual":
        error = float((logits.float() - donor_logits.float()).abs().max().cpu())
        if not torch.isfinite(logits).all() or not error <= 1e-5:
            raise RuntimeError(f"Positive transfer did not reproduce donor logits: {error}")
    trace = {
        "step": step, "recipient_input_token_id": int(token.item()),
        "donor_input_token_id": int(token.item()) if donor_before is not None else None,
        "recipient_cache_length_before": position, "donor_cache_length_before": donor_before,
        "patch_applied": active, "verified_hook_calls": {str(layer): int(active) for layer, _ in pairs},
        "verified_final_norm_calls": {"donor": 1, "recipient": 1} if action == "residual" else {},
        "donor_logit_max_abs_error": error,
        # A reference query was captured once, not forwarded again for each recipient branch.
        "resample_reference_id": resample.reference_id if action == "resample" and active else None,
        "resample_query_token_id": resample.input_token_id if action == "resample" and active else None,
        "resample_query_position": resample.position if action == "resample" and active else None,
    }
    return logits, recipient_past, donor_past, trace


@torch.no_grad()
def continue_branch(model, tokenizer, start, stepper, max_new_tokens, forced_ids=None):
    logits, recipient_start, donor_start, query_trace = start
    recipient, donor = copy.deepcopy(recipient_start), copy.deepcopy(donor_start)
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    tokens, logprobs, trace = [], [], [query_trace]
    stop = "teacher_forced" if forced_ids is not None else "token_budget"
    budget = len(forced_ids) if forced_ids is not None else max_new_tokens
    for index in range(budget):
        chosen = forced_ids[index] if forced_ids is not None else int(logits.argmax(-1).item())
        tokens.append(chosen)
        logprobs.append(float(logits.float().log_softmax(-1)[0, chosen].cpu()))
        if forced_ids is None and chosen in eos:
            stop = "eos"
            break
        if index + 1 < budget:
            token = torch.tensor([[chosen]], device=model.device)
            logits, recipient, donor, event = stepper(token, recipient, donor, index + 1)
            trace.append(event)
    return {"token_ids": tokens, "token_logprobs": logprobs, "steps": trace, "stop": stop}


@torch.no_grad()
def evaluate(model, tokenizer, layout, recipient, max_new_tokens, action="baseline", pairs=None,
             scope="none", donor=None, resample=None):
    pairs = list(map(tuple, pairs or []))
    if action in ("baseline", "patch"):
        if resample is not None or (action == "baseline" and (pairs or donor is not None or scope != "none")):
            raise ValueError("Invalid baseline or patch arguments")
        if action == "patch" and scope not in ("prompt_only", "prompt_and_decode"):
            raise ValueError("Invalid specificity patch scope")
        return temporal_evaluate(model, tokenizer, layout, recipient, max_new_tokens, donor, pairs, scope)
    if action not in ("zero", "resample", "residual"):
        raise ValueError("Unknown specificity intervention")
    if action == "residual":
        if scope != "prompt_and_decode" or donor is None or pairs or resample is not None:
            raise ValueError("Positive transfer requires an aligned donor and no selected heads")
        if (donor.length != recipient.length or not torch.equal(donor.last_token, recipient.last_token)
                or donor.candidate_ids != recipient.candidate_ids or donor.answers != recipient.answers):
            raise ValueError("Positive donor/recipient alignment mismatch")
    elif scope != "prompt_only" or not pairs or donor is not None or (action == "resample") != (resample is not None):
        raise ValueError("Ablation/resampling requires query-only heads and the correct replacement source")
    if max_new_tokens < 1 or recipient.length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Invalid generation budget")
    prepared = [recipient] + ([donor] if donor is not None else [])
    if any(cache_seq_len(item.past) != item.length - 1 for item in prepared):
        raise RuntimeError("A prepared cache was mutated")

    def stepper(token, recipient_past, donor_past, index):
        return intervention_step(model, layout, token, recipient_past, donor_past, action, pairs,
                                 index, recipient.length, resample)

    with use_attn_impl(model, "sdpa"):
        start = stepper(recipient.last_token, copy.deepcopy(recipient.past),
                        copy.deepcopy(donor.past) if donor is not None else None, 0)
        scores = {}
        for label, ids in recipient.candidate_ids.items():
            score = continue_branch(model, tokenizer, start, stepper, max_new_tokens, ids)
            score.update(sum_logprob=sum(score["token_logprobs"]), mean_logprob=sum(score["token_logprobs"]) / len(ids))
            scores[label] = score
        generated = continue_branch(model, tokenizer, start, stepper, max_new_tokens)
    if any(cache_seq_len(item.past) != item.length - 1 for item in prepared):
        raise RuntimeError("Evaluation mutated a prepared cache")
    if cache_seq_len(start[1]) != recipient.length or (start[2] is not None and cache_seq_len(start[2]) != donor.length):
        raise RuntimeError("Continuation mutated a shared post-query cache")
    return {
        "candidate_scores": scores, **{f"{label}_mean_logprob": value["mean_logprob"] for label, value in scores.items()},
        "answer_logprob_margin": scores["clean"]["sum_logprob"] - scores["corrupt"]["sum_logprob"],
        "clean_decoy_logprob_margin": scores["clean"]["sum_logprob"] - scores["decoy"]["sum_logprob"],
        "first_clean_logprob": scores["clean"]["token_logprobs"][0],
        "first_answer_logprob_margin": scores["clean"]["token_logprobs"][0] - scores["corrupt"]["token_logprobs"][0],
        "generated_token_ids": generated["token_ids"], "generated_token_logprobs": generated["token_logprobs"],
        "generation_steps": generated["steps"], "generation_stop": generated["stop"],
        **output_metrics(tokenizer.decode(generated["token_ids"], skip_special_tokens=True), generated["token_ids"],
                         recipient.answers, recipient.candidate_ids),
    }
