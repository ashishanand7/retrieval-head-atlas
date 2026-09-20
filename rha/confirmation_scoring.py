"""Strict query-step interventions with likelihood and free-generation readouts."""

import copy
from contextlib import contextmanager, nullcontext

import torch

from rha.confirmation import answer_ids, generation_metrics
from rha.generation import cache_seq_len, one_step_logits, prefill_build_kv
from rha.interventions import HeadAblator, HeadPatcher, OProjInputCacher, spec_from_pairs
from rha.modeling import use_attn_impl


@contextmanager
def checked_intervention(layout, pairs, donor=None):
    spec = spec_from_pairs(pairs)
    if not spec or len(pairs) != len(set(map(tuple, pairs))):
        raise ValueError("Intervention must specify distinct heads")
    for layer, heads in spec.items():
        if not 0 <= layer < len(layout.o_projs) or any(not 0 <= head < layout.num_heads for head in heads):
            raise ValueError("Intervention head is outside model layout")
        if donor is not None:
            if layer not in donor or tuple(donor[layer].shape) != (1, 1, layout.hidden_size):
                raise ValueError(f"Missing or incompatible donor cache for layer {layer}")
    calls = {layer: 0 for layer in spec}
    handles = []

    def observer(layer):
        def hook(_module, inputs):
            if tuple(inputs[0].shape) != (1, 1, layout.hidden_size):
                raise ValueError("Expected a single query-token attention-output slice")
            calls[layer] += 1
        return hook

    try:
        for layer in spec:
            handles.append(layout.o_projs[layer].register_forward_pre_hook(observer(layer)))
        context = HeadAblator(layout, spec) if donor is None else HeadPatcher(layout, donor, spec)
        with context:
            yield
        if any(count != 1 for count in calls.values()):
            raise RuntimeError(f"Intervention did not fire exactly once at each selected layer: {calls}")
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def score_continuation(model, logits, past, token_ids):
    # Each branch owns its KV cache; Transformers caches may mutate in place.
    branch_past = copy.deepcopy(past)
    token_logprobs = []
    for index, token_id in enumerate(token_ids):
        token_logprobs.append(float(logits.float().log_softmax(-1)[0, token_id].cpu()))
        if index + 1 < len(token_ids):
            token = torch.tensor([[token_id]], device=model.device)
            logits, branch_past = one_step_logits(model, token, branch_past)
    return {
        "mean_logprob": sum(token_logprobs) / len(token_logprobs),
        "sum_logprob": sum(token_logprobs), "first_logprob": token_logprobs[0],
    }


@torch.no_grad()
def greedy_continuation(model, tokenizer, logits, past, max_new_tokens):
    branch_past = copy.deepcopy(past)
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if configured is None:
        configured = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(configured if isinstance(configured, (list, tuple)) else [configured])
    tokens, stop = [], "token_budget"
    for index in range(max_new_tokens):
        token_id = int(logits.argmax(-1).item())
        tokens.append(token_id)
        if token_id in eos_ids:
            stop = "eos"
            break
        if index + 1 < max_new_tokens:
            logits, branch_past = one_step_logits(
                model, torch.tensor([[token_id]], device=model.device), branch_past,
            )
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens, stop


@torch.no_grad()
def evaluate(model, tokenizer, layout, prompt, gold, other, max_len, chunk_size,
             max_new_tokens, pairs=None, donor=None, capture=False):
    length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    limit = getattr(model.config, "max_position_embeddings", max_len + max_new_tokens)
    if length > max_len or length < 2:
        raise ValueError("Prompt would be truncated or is too short")
    if chunk_size < 1 or max_new_tokens < 1:
        raise ValueError("Chunk and generation budgets must be positive")
    clean_ids, corrupt_ids = answer_ids(tokenizer, prompt, gold), answer_ids(tokenizer, prompt, other)
    if length + max(max_new_tokens, len(clean_ids), len(corrupt_ids)) > limit:
        raise ValueError("Prompt plus continuation exceeds model position limit")
    past, input_ids = prefill_build_kv(model, tokenizer, prompt, max_len, chunk_size)
    if input_ids.shape[1] != length or cache_seq_len(past) != length - 1:
        raise RuntimeError("Prefill lost tokens or returned an unexpected cache length")
    intervention = checked_intervention(layout, pairs, donor) if pairs else nullcontext()
    cacher = OProjInputCacher(layout) if capture else nullcontext()
    with use_attn_impl(model, "sdpa"):
        with cacher as captured, intervention:
            logits, past = one_step_logits(model, input_ids[:, -1:], past)
        # Interventions are already removed for both teacher forcing and generation.
        expected_length = cache_seq_len(past)
        clean_score = score_continuation(model, logits, past, clean_ids)
        corrupt_score = score_continuation(model, logits, past, corrupt_ids)
        text, token_ids, stop = greedy_continuation(model, tokenizer, logits, past, max_new_tokens)
        if cache_seq_len(past) != expected_length:
            raise RuntimeError("A continuation branch mutated the shared query cache")
    metrics = {
        "clean_mean_logprob": clean_score["mean_logprob"],
        "corrupt_mean_logprob": corrupt_score["mean_logprob"],
        "answer_logprob_margin": clean_score["sum_logprob"] - corrupt_score["sum_logprob"],
        "first_clean_logprob": clean_score["first_logprob"],
        "first_answer_logprob_margin": clean_score["first_logprob"] - corrupt_score["first_logprob"],
        "generated_token_ids": token_ids, "generation_stop": stop,
        **generation_metrics(text, gold, other),
    }
    return metrics, captured.cache if capture else None
