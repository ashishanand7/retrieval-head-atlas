from contextlib import nullcontext
from typing import Optional

import torch

from rha.modeling import use_attn_impl


def cache_seq_len(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    first = past[0]
    key = first[0] if isinstance(first, (tuple, list)) else first.key
    return int(key.shape[-2])


@torch.no_grad()
def prefill_build_kv(model, tokenizer, prompt: str, max_len: int, chunk_size: int):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]
    if input_ids.shape[1] < 2:
        raise ValueError("Prompt must tokenize to at least two tokens.")

    pre_ids = input_ids[:, :-1]
    past = None

    with use_attn_impl(model, "sdpa"):
        for start in range(0, pre_ids.shape[1], chunk_size):
            chunk = pre_ids[:, start:start + chunk_size]
            past_len = cache_seq_len(past)
            attention_mask = torch.ones(
                (chunk.shape[0], past_len + chunk.shape[1]),
                device=model.device,
                dtype=torch.long,
            )
            position_ids = torch.arange(
                past_len,
                past_len + chunk.shape[1],
                device=model.device,
                dtype=torch.long,
            ).unsqueeze(0)
            kwargs = dict(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            try:
                out = model(**kwargs, position_ids=position_ids)
            except TypeError:
                out = model(**kwargs)
            past = out.past_key_values

    return past, input_ids


@torch.no_grad()
def one_step_logits(model, token_id: torch.Tensor, past):
    past_len = cache_seq_len(past)
    step_len = token_id.shape[1]
    attention_mask = torch.ones(
        (token_id.shape[0], past_len + step_len),
        device=model.device,
        dtype=torch.long,
    )
    position_ids = torch.arange(
        past_len,
        past_len + step_len,
        device=model.device,
        dtype=torch.long,
    ).unsqueeze(0)
    kwargs = dict(
        input_ids=token_id,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    try:
        out = model(**kwargs, position_ids=position_ids)
    except TypeError:
        out = model(**kwargs)
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def mean_gold_logprob(
    model,
    tokenizer,
    prompt: str,
    gold: str,
    max_len: int,
    chunk_size: int,
    intervention: Optional[object] = None,
    intervention_scope: str = "all",
) -> float:
    gold_ids = tokenizer(gold, add_special_tokens=False).input_ids
    if not gold_ids:
        return float("nan")

    past, input_ids = prefill_build_kv(model, tokenizer, prompt, max_len=max_len, chunk_size=chunk_size)
    last_tok = input_ids[:, -1:]

    def score_loop(apply_ctx_to_all: bool) -> float:
        total = 0.0
        ctx = intervention if (intervention is not None and apply_ctx_to_all) else nullcontext()
        with ctx:
            logits, cur_past = one_step_logits(model, last_tok, past)
            for token_id in gold_ids:
                total += float(torch.log_softmax(logits, dim=-1)[0, token_id].detach().cpu())
                next_tok = torch.tensor([[token_id]], device=model.device)
                logits, cur_past = one_step_logits(model, next_tok, cur_past)
        return total / len(gold_ids)

    with use_attn_impl(model, "sdpa"):
        if intervention is None:
            return score_loop(apply_ctx_to_all=False)
        if intervention_scope == "all":
            return score_loop(apply_ctx_to_all=True)
        if intervention_scope == "query":
            with intervention:
                logits, cur_past = one_step_logits(model, last_tok, past)
            total = 0.0
            for token_id in gold_ids:
                total += float(torch.log_softmax(logits, dim=-1)[0, token_id].detach().cpu())
                next_tok = torch.tensor([[token_id]], device=model.device)
                logits, cur_past = one_step_logits(model, next_tok, cur_past)
            return total / len(gold_ids)
        raise ValueError(f"Unknown intervention_scope: {intervention_scope}")
