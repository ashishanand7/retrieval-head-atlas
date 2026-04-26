# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: RHA (uv venv)
#     language: python
#     name: rha-venv
# ---

# %%
# Cell 1 
import os, sys, json, random
import numpy as np
import torch

# Cache: big local path reduces repeated downloads
os.environ["HF_HOME"] = os.path.expanduser("~/SageMaker/hf-cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

def set_seeds(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seeds(1234)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))


# %%
# Cell 2 
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

cfg = yaml.safe_load(open("../config.yaml"))

MODEL_ID = cfg["model_id"]
DTYPE = torch.float16 if cfg["precision"] in ("float16","fp16") else "auto"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
)

print("Loaded:", MODEL_ID)
print("n_layers:", getattr(model.config, "num_hidden_layers", None))
print("n_heads:", getattr(model.config, "num_attention_heads", None))
print("max_position_embeddings:", getattr(model.config, "max_position_embeddings", None))


# %%
# Cell 4 
def set_attn_impl(model, mode="flash_attention_2"):
    # Prefer flash-attn if available; fall back to sdpa; avoid 'eager' except when explicitly needed
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(mode)
        else:
            # newer HF uses private attr on config
            model.config._attn_implementation = mode
        print("Attention impl set to:", mode)
    except Exception as e:
        print("Could not set to", mode, "-> falling back to 'sdpa'. Error:", e)
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("sdpa")
        else:
            model.config._attn_implementation = "sdpa"
        print("Attention impl set to: sdpa")

# We ONLY need 'eager' when we explicitly want attention weights (e.g., Day 3).
# For regular generation/smoke tests, use flash or sdpa:
set_attn_impl(model, "flash_attention_2")  # will auto-fallback to sdpa if flash-attn isn't available


# %%
# Cell 5 
import platform, json, transformers
meta = {
    "model_id": MODEL_ID,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "device": str(model.device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
}
os.makedirs("../artifacts", exist_ok=True)
json.dump(meta, open("../artifacts/meta_day1.json","w"), indent=2)
meta


# %% [markdown]
# ### Probe Harness

# %%
# Cell 7 
import os, json, math, random, numpy as np, torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
from tqdm import trange, tqdm

# Same cache as Day 1
os.environ["HF_HOME"] = os.path.expanduser("~/SageMaker/hf-cache")
ART_DIR = os.path.expanduser("~/SageMaker/retrieval-head-atlas/artifacts_new")
os.makedirs(ART_DIR, exist_ok=True)

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# %%
# Cell 19
# ---------------------------
# Day 3 — Long-range sweep (report-grade)
# - Harder prompt (deterministic decoy + explicit instruction)
# - Copy-paste event = detect & copied_first_token (ICLR-style)
# - Head score = P(detect | copied_first) = event_rate / copied_first_rate
# ---------------------------

import os, re, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

model.eval()

# --- make sure these exist even if you delete legacy cells ---
ART_DIR = os.path.expanduser("~/SageMaker/retrieval-head-atlas/artifacts_new")
os.makedirs(ART_DIR, exist_ok=True)

# robust dims
n_layers = int(getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0)))
n_heads  = int(getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 0)))
assert n_layers > 0 and n_heads > 0, "Could not read n_layers/n_heads from model.config"

NEEDLE_START = "[NEEDLE_START]"
NEEDLE_END   = "[NEEDLE_END]"

# ---------------------------
# 0) Small helpers
# ---------------------------

_WORDS = [
    "lorem","ipsum","dolor","sit","amet","consectetur","adipiscing","elit",
    "sed","do","eiusmod","tempor","incididunt","ut","labore","et","dolore",
    "magna","aliqua","enim","ad","minim","veniam","quis","nostrud",
    "exercitation","ullamco","laboris","nisi","ut","aliquip","ex","ea",
    "commodo","consequat","duis","aute","irure","dolor","in","reprehenderit",
    "voluptate","velit","esse","cillum","dolore","eu","fugiat","nulla",
    "pariatur","excepteur","sint","occaecat","cupidatat","non","proident"
]

def _seeded_decoy_6digits(seed: int) -> int:
    # deterministic 6-digit decoy based ONLY on seed (so clean/corrupt pairs share it)
    x = (int(seed) * 1103515245 + 12345) & 0x7FFFFFFF
    return int(x % 900000) + 100000

def find_subsequence(haystack: List[int], needle: List[int]) -> Tuple[int, int]:
    """Return (start, end) for first exact match of needle in haystack; (-1,-1) if none."""
    if len(needle) == 0 or len(needle) > len(haystack):
        return -1, -1
    # naive is fine for our lengths (needle is tiny)
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i, i + len(needle)
    return -1, -1

def char_span_to_token_span(offsets, char_start: int, char_end: int):
    """
    offsets: list[(s,e)] for each token.
    Return (tok_start, tok_end_excl) covering the char span [char_start, char_end).
    """
    tok_start = None
    tok_end = None

    for i, (s, e) in enumerate(offsets):
        # token overlaps the char span?
        if e > char_start and s < char_end:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1

    if tok_start is None or tok_end is None:
        return -1, -1
    return tok_start, tok_end

def find_marked_answer_token_span(prompt: str, gold_answer: str, ids_list, offsets):
    """
    Find gold_answer inside NEEDLE markers using string search (robust to multiple mentions),
    then map to token span via offsets.
    Returns (ans_s0, ans_s1) token indices.
    """

    # Prefer the actual marker block in the document:
    # use the *last* NEEDLE_START and the first NEEDLE_END after it.
    ms = prompt.rfind(NEEDLE_START)
    if ms == -1:
        return -1, -1

    me = prompt.find(NEEDLE_END, ms + len(NEEDLE_START))
    if me == -1 or me <= ms:
        return -1, -1

    region_start = ms + len(NEEDLE_START)
    region_end = me

    # Locate digits inside that region
    a0 = prompt.find(str(gold_answer), region_start, region_end)
    if a0 == -1:
        # fallback: if something went odd, try regex inside region
        region_txt = prompt[region_start:region_end]
        m = re.search(re.escape(str(gold_answer)), region_txt)
        if not m:
            return -1, -1
        a0 = region_start + m.start()

    a1 = a0 + len(str(gold_answer))

    # Map char span -> token span using offsets
    ans_s0, ans_s1 = char_span_to_token_span(offsets, a0, a1)
    return ans_s0, ans_s1



# ---------------------------
# 1) Prompt builder (harder & deterministic decoy)
# ---------------------------

def _build_prompt_positioned(
    total_filler_words: int,
    needle_frac: float,
    secret: int,
    seed: int
) -> str:
    """
    Build a long prompt with a marked 'needle' containing secret, plus a decoy 6-digit code.
    Decoy depends only on seed so paired prompts share it.
    """
    rng = random.Random(int(seed))

    # deterministic decoy 6 digits (seed-only)
    decoy = _seeded_decoy_6digits(seed)

    # split filler into prefix/suffix
    n_pre = int(round(total_filler_words * float(needle_frac)))
    n_pre = max(0, min(total_filler_words, n_pre))
    n_suf = int(total_filler_words - n_pre)

    prefix = " ".join(rng.choice(_WORDS) for _ in range(n_pre))
    suffix = " ".join(rng.choice(_WORDS) for _ in range(n_suf))

    instructions = (
        "You will answer a question about the document below.\n"
        "The document may contain decoy numbers.\n"
        "ONLY trust the number inside the NEEDLE markers.\n"
        "Answer with EXACTLY six digits and nothing else.\n\n"
    )

    decoy_line = f"Unrelated archival code (decoy): {decoy}.\n"
    needle_line = f"{NEEDLE_START} The true secret number is {int(secret)}. {NEEDLE_END}\n"

    question = "Q: What is the true secret number? A:"

    doc = (
        instructions
        + prefix + "\n"
        + decoy_line
        + needle_line
        + suffix + "\n"
        + question
    )
    return doc

def calibrate_total_filler_words_for_setting(
    tok,
    target_tokens: int,
    needle_frac: float,
    secret: int,
    seed: int,
    margin_tokens: int = 32,
    max_tok_cap: int = 12000,
    max_iters: int = 40
) -> int:
    """
    Find total_filler_words such that tokenized prompt length ~= target_tokens (minus margin).
    Uses a simple binary search over word count.
    """
    target = int(min(max_tok_cap, max(256, target_tokens)))
    desired = max(1, target - int(margin_tokens))

    lo, hi = 0, 200000  # generous ceiling
    best = None
    for _ in range(max_iters):
        mid = (lo + hi) // 2
        prompt = _build_prompt_positioned(mid, needle_frac, secret, seed)
        n_tok = int(tok(prompt, add_special_tokens=False).input_ids.__len__())
        if abs(n_tok - desired) <= 3:
            best = mid
            break
        if n_tok < desired:
            lo = mid + 1
        else:
            hi = mid - 1
        best = mid

    return int(best if best is not None else lo)
    
def make_prompt_for_setting(
    tok,
    target_tokens: int,
    needle_frac: float,
    total_filler_words: int,
    secret: int,
    seed: int,
    margin_tokens: int = 32,
    max_tok_cap: int = 12000,
):
    """
    Returns (prompt, gold_answer_str). Regenerates if token length > cap to avoid decode-truncation.
    """
    tw = int(total_filler_words)

    for _ in range(6):
        prompt = _build_prompt_positioned(tw, needle_frac, secret, seed)

        # hard requirement: markers exist in raw string
        if NEEDLE_START not in prompt or NEEDLE_END not in prompt:
            tw = max(50, int(tw * 0.95))
            continue

        ids = tok(prompt, add_special_tokens=False).input_ids
        if len(ids) <= max_tok_cap:
            return prompt, str(int(secret))

        # too long → shrink and retry
        tw = max(50, int(tw * 0.95))

    # final fallback
    return prompt, str(int(secret))

# ---------------------------
# 2) Attn-implementation switching helpers
# ---------------------------

def _get_attn_impl(model) -> Optional[str]:
    # transformers vary: sometimes _attn_implementation, sometimes attn_implementation
    if hasattr(model.config, "_attn_implementation"):
        return getattr(model.config, "_attn_implementation")
    return getattr(model.config, "attn_implementation", None)

def _set_attn_impl(model, val: str):
    if hasattr(model.config, "_attn_implementation"):
        setattr(model.config, "_attn_implementation", val)
    else:
        setattr(model.config, "attn_implementation", val)

def _cache_seq_len(past) -> int:
    """
    Try to infer cached length from various cache formats.
    """
    if past is None:
        return 0
    # tuple-of-layer tuples: ((k,v), (k,v), ...)
    if isinstance(past, (tuple, list)) and len(past) > 0:
        first = past[0]
        if isinstance(first, (tuple, list)) and len(first) > 0:
            k = first[0]
            if hasattr(k, "shape") and len(k.shape) >= 3:
                return int(k.shape[-2])  # (bs, heads, seq, dim) or similar
    # modern Cache objects sometimes have get_seq_length
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return 0

@torch.no_grad()
def prefill_build_kv(
    model,
    tok,
    prompt: str,
    max_len: int = 12000,
    chunk_size: int = 512
):
    """
    Chunked prefill under SDPA/Flash, leaving the last token for the eager step.
    Returns (past, input_ids_full) where input_ids_full includes the last prompt token.
    """
    old_impl = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]

    # leave last token for eager step
    pre_ids = input_ids[:, :-1]
    past = None

    # chunk through pre_ids
    for i in range(0, pre_ids.shape[1], chunk_size):
        chunk = pre_ids[:, i:i+chunk_size]
        past_len = _cache_seq_len(past)
        pos = torch.arange(past_len, past_len + chunk.shape[1], device=model.device).unsqueeze(0)
        attn_mask = torch.ones((1, past_len + chunk.shape[1]), device=model.device, dtype=torch.long)

        outs = model(
            input_ids=chunk,
            attention_mask=attn_mask,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = outs.past_key_values

    _set_attn_impl(model, old_impl if old_impl is not None else "sdpa")
    return past, input_ids

@torch.no_grad()
def one_step_eager_with_attn(model, last_tok, past):
    """
    Process ONE token (last_tok) with output_attentions=True under eager attention.
    Returns (next_tok_id, logits, attentions_per_layer, past_after)
    """
    old_impl = _get_attn_impl(model)
    _set_attn_impl(model, "eager")

    past_len = _cache_seq_len(past)
    pos = torch.tensor([[past_len]], device=model.device, dtype=torch.long)
    attn_mask = torch.ones((1, past_len + 1), device=model.device, dtype=torch.long)

    out = model(
        input_ids=last_tok,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=past,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )

    logits = out.logits[:, -1, :]        # [1, vocab]
    next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]
    attns = out.attentions              # tuple of length n_layers, each [1, heads, 1, K]
    past2 = out.past_key_values

    _set_attn_impl(model, old_impl if old_impl is not None else "sdpa")
    return next_tok, logits, attns, past2

@torch.no_grad()
def one_step_sdpa_no_attn(model, tok_ids, past):
    """
    Process ONE token under sdpa, no attentions.
    """
    old_impl = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    past_len = _cache_seq_len(past)
    pos = torch.tensor([[past_len]], device=model.device, dtype=torch.long)
    attn_mask = torch.ones((1, past_len + 1), device=model.device, dtype=torch.long)

    out = model(
        input_ids=tok_ids,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=past,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    logits = out.logits[:, -1, :]
    past2 = out.past_key_values

    _set_attn_impl(model, old_impl if old_impl is not None else "sdpa")
    return logits, past2


_RE6 = re.compile(r"(\d{6})")
def extract_6digits(text: str) -> Optional[str]:
    m = _RE6.search(text)
    return m.group(1) if m else None

# ---------------------------
# 3) Day3 scoring (copy event)
# ---------------------------
@torch.no_grad()
def greedy_until_6digits(model, tok, first_tok, past_after, max_steps: int = 32):
    """
    Continue greedy decoding starting from (first_tok, past_after) until we see a 6-digit substring
    or we hit max_steps. Returns (gen_ids, gen_txt).
    IMPORTANT: does not run any “probe” decode that could mutate cache before the main decode.
    """
    gen_ids = [int(first_tok.item())]
    txt = tok.decode(gen_ids)

    if re.search(r"\d{6}", txt):
        return gen_ids, txt

    cur = first_tok
    past = past_after
    for _ in range(max_steps - 1):
        logits, past = one_step_sdpa_no_attn(model, cur, past)
        cur = torch.argmax(logits, dim=-1, keepdim=True)
        gen_ids.append(int(cur.item()))
        txt = tok.decode(gen_ids)
        if re.search(r"\d{6}", txt):
            break

    return gen_ids, txt

@torch.no_grad()
def score_prompt_day3(prompt: str, gold_answer: str, tau: float = 0.15, max_new_tokens: int = 32) -> Dict:
    """
    Robust marker finding using prompt string + offset_mapping.
    Copy-paste event uses:
      copied_first = does the model start its answer with the gold 6-digit number (ignoring whitespace)?
    Retrieval uses:
      extracted = first 6-digit anywhere in decoded answer prefix
    """

    # tokenize WITH offsets so we can map char spans → token spans reliably
    enc = tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    input_ids = enc["input_ids"].to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    ids_list = input_ids[0].tolist()

    # KV cache (prefill) leaving last token for eager attention readout
    past, _input_ids_full = prefill_build_kv(model, tok, prompt, max_len=12000, chunk_size=512)

    # Find answer token span inside the last marker block (string-based + offsets)
    ans_s0, ans_s1 = find_marked_answer_token_span(prompt, gold_answer, ids_list, offsets)
    if ans_s0 == -1:
        ms = prompt.rfind(NEEDLE_START)
        me = prompt.find(NEEDLE_END, ms + len(NEEDLE_START)) if ms != -1 else -1
        region = prompt[ms:me+len(NEEDLE_END)] if (ms != -1 and me != -1) else "NO_REGION"
        raise RuntimeError(f"Could not locate gold answer span via offsets. Region:\n{region}")

    # First generation step with attentions (query = last prompt token)
    last_tok = input_ids[:, -1:].to(model.device)
    first_tok, _logits, attns_first, past_after = one_step_eager_with_attn(model, last_tok, past)

    # Main greedy decode (single path) up to max_new_tokens OR until 6 digits appear
    gen_ids, gen_txt = greedy_until_6digits(model, tok, first_tok, past_after, max_steps=int(max_new_tokens))

    gold = str(gold_answer)

    # extracted: first 6-digit substring anywhere in generated prefix
    extracted = extract_6digits(gen_txt)
    retrieved = int(extracted == gold)

    # copied_first: does the answer start with the gold digits (allow whitespace before digits)
    m0 = re.match(r"\s*(\d{6})", gen_txt)
    copied_first = int(m0 is not None and m0.group(1) == gold)

    # Compute per-head attention mass on the answer span (FIRST step attentions)
    masses = np.zeros((n_layers, n_heads), dtype=np.float64)
    hits   = np.zeros((n_layers, n_heads), dtype=np.int32)

    for L in range(n_layers):
        aL = attns_first[L][0].detach().float().cpu().numpy()   # [heads, 1, K]
        span_mass = aL[:, 0, ans_s0:ans_s1].sum(axis=-1)
        masses[L, :] = span_mass
        hits[L, :] = (span_mass >= float(tau)).astype(np.int32)

    q_pos = len(ids_list) - 1
    distance_tokens = int(q_pos - ans_s1)

    return {
        "masses": masses,
        "hits": hits,
        "copied_first": copied_first,
        "retrieved": retrieved,
        "extracted": extracted,
        "gen_txt": gen_txt,
        "distance_tokens": distance_tokens,
        "prompt_len_tokens": int(len(ids_list)),
        "ans_s0": int(ans_s0),
        "ans_s1": int(ans_s1),
    }

def plot_heatmap(mat: np.ndarray, title: str, out_png: str, vmin: float = 0.0, vmax: float = 1.0):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, label="value")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# 4) Sweep runner
# ---------------------------

@dataclass
class Day3Config:
    lengths: Tuple[int, ...] = (8192,)
    needle_fracs: Tuple[float, ...] = (0.10, 0.90)  # near vs far
    n_prompts_per_setting: int = 40
    tau: float = 0.15
    margin_tokens: int = 32
    max_len_encode: int = 12000
    chunk_size: int = 512
    seed_base: int = 7777
    out_prefix: str = "day3"

def run_day3_sweep(cfg: Day3Config):
    settings = []
    results_by_setting = {}

    for L in cfg.lengths:
        for frac in cfg.needle_fracs:
            key = f"L{int(L)}_p{float(frac):.2f}"

            # calibrate once per setting
            tw = calibrate_total_filler_words_for_setting(
                tok=tok,
                target_tokens=int(L),
                needle_frac=float(frac),
                secret=555555,
                seed=0,
                margin_tokens=cfg.margin_tokens,
                max_tok_cap=cfg.max_len_encode,
            )

            detect_cnt = np.zeros((n_layers, n_heads), dtype=np.int32)
            event_cnt  = np.zeros((n_layers, n_heads), dtype=np.int32)
            mass_sum   = np.zeros((n_layers, n_heads), dtype=np.float64)

            copied_ok = 0
            retrieved_ok = 0
            dist_sum = 0.0
            plen_sum = 0.0

            rng = np.random.default_rng(cfg.seed_base + int(L) * 1000 + int(frac * 10000))

            for i in tqdm(range(cfg.n_prompts_per_setting), desc=f"Day3 {key}"):
                secret = int(rng.integers(100000, 999999))
                seed = int(cfg.seed_base + i + int(L) * 13 + int(frac * 10000))

                prompt, gold = make_prompt_for_setting(
                    tok=tok,
                    target_tokens=int(L),
                    needle_frac=float(frac),
                    total_filler_words=int(tw),
                    secret=secret,
                    seed=seed,
                    margin_tokens=cfg.margin_tokens,
                    max_tok_cap=cfg.max_len_encode,
                )

                # out = score_prompt_day3(prompt, gold, tau=cfg.tau, max_new_tokens=8)
                out = score_prompt_day3(prompt, gold, tau=cfg.tau, max_new_tokens=32)

                detect_cnt += out["hits"]
                mass_sum   += out["masses"]

                copied_ok += int(out["copied_first"])
                retrieved_ok += int(out["retrieved"])
                dist_sum += float(out["distance_tokens"])
                plen_sum += float(out["prompt_len_tokens"])

                # copy-paste event: detect & copied_first
                if out["copied_first"] == 1:
                    event_cnt += out["hits"]

            N = float(cfg.n_prompts_per_setting)
            detect_rate = detect_cnt / max(1.0, N)
            event_rate  = event_cnt / max(1.0, N)
            avg_mass    = mass_sum / max(1.0, N)

            copied_first_rate = float(copied_ok / max(1.0, N))
            retrieval_rate    = float(retrieved_ok / max(1.0, N))

            # head score: P(detect | copied_first)
            if copied_first_rate > 0:
                retrieval_score = event_rate / copied_first_rate
            else:
                retrieval_score = np.zeros_like(event_rate)

            settings.append({
                "setting": key,
                "length": int(L),
                "needle_frac": float(frac),
                "n": int(cfg.n_prompts_per_setting),
                "tau": float(cfg.tau),
                "retrieval_rate": retrieval_rate,
                "copied_first_rate": copied_first_rate,
                "avg_distance_tokens": float(dist_sum / max(1.0, N)),
                "avg_prompt_len_tokens": float(plen_sum / max(1.0, N)),
            })

            results_by_setting[key] = {
                "detect_rate": detect_rate.tolist(),
                "event_rate": event_rate.tolist(),
                "avg_mass": avg_mass.tolist(),
                "retrieval_score": retrieval_score.tolist(),
                "retrieval_rate": retrieval_rate,
                "copied_first_rate": copied_first_rate,
                "avg_distance_tokens": float(dist_sum / max(1.0, N)),
                "avg_prompt_len_tokens": float(plen_sum / max(1.0, N)),
            }

            # quick per-setting plots (optional but nice for artifacts)
            plot_heatmap(
                event_rate,
                f"{key}: copy-event rate (detect & copied_first), tau={cfg.tau}",
                os.path.join(ART_DIR, f"{cfg.out_prefix}_{key}_event_heatmap.png"),
                vmin=0.0, vmax=1.0
            )
            plot_heatmap(
                retrieval_score,
                f"{key}: P(detect | copied_first) head score",
                os.path.join(ART_DIR, f"{cfg.out_prefix}_{key}_score_heatmap.png"),
                vmin=0.0, vmax=1.0
            )

    out = {
        "model_id": getattr(model, "name_or_path", "unknown"),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "cfg": cfg.__dict__,
        "settings": results_by_setting
    }

    json_path = os.path.join(ART_DIR, f"{cfg.out_prefix}_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f)
    print("Saved:", json_path)

    df_setting_summary = pd.DataFrame(settings)
    csv_path = os.path.join(ART_DIR, f"{cfg.out_prefix}_setting_summary.csv")
    df_setting_summary.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    return out, df_setting_summary

cfg3 = Day3Config()
day3_results, df_setting_summary = run_day3_sweep(cfg3)
df_setting_summary


# %%
# Cell 19.5
# ---------------------------
# Day3 saturation diagnostics (head-level)
# ---------------------------
import numpy as np

def _mat(day3_results, setting_key, field):
    x = day3_results["settings"][setting_key][field]
    return np.array(x, dtype=float)

def summarize_setting(day3_results, setting_key):
    # Try retrieval_score first; fall back to event_rate if needed
    if "retrieval_score" in day3_results["settings"][setting_key]:
        M = _mat(day3_results, setting_key, "retrieval_score")
        name = "retrieval_score"
    else:
        M = _mat(day3_results, setting_key, "event_rate")
        name = "event_rate"

    flat = M.reshape(-1)
    print(f"\n== {setting_key} ({name}) ==")
    print("shape:", M.shape)
    print("mean:", float(flat.mean()), "std:", float(flat.std()))
    print("p50:", float(np.quantile(flat, 0.50)),
          "p90:", float(np.quantile(flat, 0.90)),
          "p99:", float(np.quantile(flat, 0.99)),
          "max:", float(flat.max()))
    print("#heads >=0.90:", int((flat >= 0.90).sum()),
          "| >=0.50:", int((flat >= 0.50).sum()),
          "| >=0.20:", int((flat >= 0.20).sum()))

    # show top 15 heads
    idx = np.argsort(-flat)[:15]
    top = []
    H = M.shape[1]
    for k in idx:
        layer = int(k // H)
        head  = int(k % H)
        top.append((layer, head, float(flat[k])))
    print("top15:", top)

# run for all settings you swept
for k in sorted(day3_results["settings"].keys()):
    summarize_setting(day3_results, k)


# %%
# Cell 20
# ==========================================
# Day 3 — Flatten results to per-head dataframe (NEW schema)
# ==========================================

import os, re
import numpy as np
import pandas as pd

def _parse_setting_key(key: str):
    """
    Expect keys like: 'L8192_p0.10'
    Returns (length:int, needle_frac:float)
    """
    m = re.match(r"^L(\d+)_p([0-9]*\.?[0-9]+)$", key.strip())
    if not m:
        raise ValueError(f"Unrecognized setting key format: {key}")
    return int(m.group(1)), float(m.group(2))

def results_to_long_df(day3_results: dict) -> pd.DataFrame:
    rows = []
    settings = day3_results["settings"]

    for key, item in settings.items():
        L, frac = _parse_setting_key(key)

        retrieval_rate = float(item.get("retrieval_rate", np.nan))
        copied_first_rate = float(item.get("copied_first_rate", np.nan))
        avg_prompt_len = float(item.get("avg_prompt_len_tokens", np.nan))
        avg_dist = float(item.get("avg_distance_tokens", np.nan))

        detect = np.array(item["detect_rate"], dtype=float)         # [layers, heads]
        event  = np.array(item["event_rate"], dtype=float)          # [layers, heads]
        score  = np.array(item["retrieval_score"], dtype=float)     # [layers, heads]
        mass   = np.array(item.get("avg_mass", np.zeros_like(score)), dtype=float)

        n_layers_local, n_heads_local = detect.shape
        for layer in range(n_layers_local):
            for head in range(n_heads_local):
                rows.append({
                    "setting": key,
                    "length": L,
                    "needle_frac": frac,
                    "layer": layer,
                    "head": head,

                    "detect_rate": float(detect[layer, head]),
                    "event_rate": float(event[layer, head]),
                    "retrieval_score": float(score[layer, head]),
                    "avg_mass": float(mass[layer, head]),

                    "retrieval_rate": retrieval_rate,
                    "copied_first_rate": copied_first_rate,
                    "avg_prompt_len_tokens": avg_prompt_len,
                    "avg_distance_tokens": avg_dist,
                })

    return pd.DataFrame(rows)

df_long = results_to_long_df(day3_results)

csv_long = os.path.join(ART_DIR, "day3_heads_long.csv")
df_long.to_csv(csv_long, index=False)
print("Saved:", csv_long)

df_long.sort_values(["event_rate", "retrieval_score"], ascending=False).head(20)


# %%
# Cell 21
# ===================================================
# Day 3 — Distance sensitivity report (far vs near)
# Keeps rank_key (for candidate selection) AND includes retrieval/copy rates for context
# ===================================================

import os
import pandas as pd
import numpy as np

def distance_sensitivity_report(
    df_long: pd.DataFrame,
    far_max: float = 0.3,
    near_min: float = 0.7
) -> pd.DataFrame:
    # head-level aggregates
    far_h = (
        df_long[df_long["needle_frac"] <= far_max]
        .groupby(["layer","head"], as_index=False)
        .agg(
            far_event=("event_rate","mean"),
            far_detect=("detect_rate","mean"),
            far_score=("retrieval_score","mean"),
        )
    )

    near_h = (
        df_long[df_long["needle_frac"] >= near_min]
        .groupby(["layer","head"], as_index=False)
        .agg(
            near_event=("event_rate","mean"),
            near_detect=("detect_rate","mean"),
            near_score=("retrieval_score","mean"),
        )
    )

    m = far_h.merge(near_h, on=["layer","head"], how="inner")

    # deltas/ratios (useful in report)
    m["delta_event_far_minus_near"] = m["far_event"] - m["near_event"]
    m["delta_score_far_minus_near"] = m["far_score"] - m["near_score"]
    m["event_ratio_far_over_near"]  = (m["far_event"] + 1e-9) / (m["near_event"] + 1e-9)
    m["score_ratio_far_over_near"]  = (m["far_score"] + 1e-9) / (m["near_score"] + 1e-9)

    # ranking key used for picking candidates
    m["rank_key"] = m["far_event"] + 0.5 * m["far_score"]

    # add setting-level context (same for all heads, but handy for tables)
    far_meta = (
        df_long[df_long["needle_frac"] <= far_max][["setting","retrieval_rate","copied_first_rate"]]
        .drop_duplicates()
    )
    near_meta = (
        df_long[df_long["needle_frac"] >= near_min][["setting","retrieval_rate","copied_first_rate"]]
        .drop_duplicates()
    )

    # If multiple far/near settings exist, average them (still just context)
    m["far_retrieval_rate_setting_mean"] = float(far_meta["retrieval_rate"].mean()) if len(far_meta) else np.nan
    m["far_copied_first_rate_setting_mean"] = float(far_meta["copied_first_rate"].mean()) if len(far_meta) else np.nan
    m["near_retrieval_rate_setting_mean"] = float(near_meta["retrieval_rate"].mean()) if len(near_meta) else np.nan
    m["near_copied_first_rate_setting_mean"] = float(near_meta["copied_first_rate"].mean()) if len(near_meta) else np.nan

    return m.sort_values(["rank_key","far_event","far_score"], ascending=False)

df_dist = distance_sensitivity_report(df_long, far_max=0.3, near_min=0.7)

dist_csv = os.path.join(ART_DIR, "day3_distance_sensitivity_heads.csv")
df_dist.to_csv(dist_csv, index=False)
print("Saved:", dist_csv)

df_dist.head(30)


# %%
# Cell 23
# ==========================================
# Day 3 — identify "far" vs "near-only" heads
# ==========================================

def classify_heads_far_vs_near(
    df_dist: pd.DataFrame,
    far_event_min: float = 0.20,           # needs to fire reasonably often when far
    far_over_near_ratio_min: float = 0.50, # far_event at least 50% of near_event
    near_event_min: float = 0.20,          # fires often when near
    far_event_max_for_near_only: float = 0.05,  # but almost never when far
    top_k: int = 32,
):
    """
    Returns two dataframes:
      - far_heads: heads that still fire when the needle is far away
      - near_only_heads: heads that basically only fire when the needle is near the question
    """

    # Heads that keep firing when the needle is far
    far_heads = df_dist[
        (df_dist["far_event"] >= far_event_min) &
        (df_dist["event_ratio_far_over_near"] >= far_over_near_ratio_min)
    ].sort_values(["far_event", "far_score"], ascending=False)

    # Heads that only fire when needle is near
    near_only_heads = df_dist[
        (df_dist["near_event"] >= near_event_min) &
        (df_dist["far_event"] <= far_event_max_for_near_only)
    ].sort_values(["near_event", "near_score"], ascending=False)

    return far_heads.head(top_k), near_only_heads.head(top_k)


far_heads, near_only_heads = classify_heads_far_vs_near(df_dist)

print("=== Heads that keep firing when the needle is FAR (e.g., 0.1 / 0.3) ===")
display(
    far_heads[[
        "layer", "head",
        "far_event", "near_event",
        "event_ratio_far_over_near",
        "far_score", "near_score"
    ]]
)

print("\n=== Heads that ONLY fire when the needle is NEAR (e.g., 0.9) ===")
display(
    near_only_heads[[
        "layer", "head",
        "far_event", "near_event",
        "event_ratio_far_over_near",
        "far_score", "near_score"
    ]]
)

# Save to CSV for later use / plotting / ablations
far_csv = os.path.join(ART_DIR, "day3_far_heads.csv")
near_only_csv = os.path.join(ART_DIR, "day3_near_only_heads.csv")
far_heads.to_csv(far_csv, index=False)
near_only_heads.to_csv(near_only_csv, index=False)

print("\nSaved far-head list to:", far_csv)
print("Saved near-only-head list to:", near_only_csv)


# %%
# Cell 24
# ============================================================
# Day 4 — Head ablations (causal tests) on long-range retrieval
# ============================================================

import os, json, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

# ---------------------------
# 0) Config
# ---------------------------

@dataclass
class Day4Config:
    # Evaluate on a small set of "far vs near" settings to keep costs sane
    lengths: Tuple[int, ...] = (8192,)
    needle_fracs: Tuple[float, ...] = (0.1, 0.9)  # far vs near
    n_prompts_per_setting: int = 80

    # Generation + cache settings (match Day 3)
    chunk_size: int = 1024
    max_len_encode: int = 16384
    margin_tokens: int = 96
    max_new_tokens: int = 14

    # Candidate head selection
    top_n_heads: int = 30
    far_max: float = 0.3
    near_min: float = 0.7

    # Output
    out_prefix: str = "day4"

cfg4 = Day4Config()
print(cfg4)


# %%
# Cell 25
# ---------------------------
# 1) Introspect model to find layers and o_proj modules
# ---------------------------

def _get_layers(model):
    # common HF layouts: model.model.layers (Llama/Qwen style), or model.transformer.h (GPT2 style)
    for root_name in ["model", "transformer"]:
        if hasattr(model, root_name):
            root = getattr(model, root_name)
            if hasattr(root, "layers"):
                return list(root.layers)
            if hasattr(root, "h"):
                return list(root.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError("Could not locate transformer layers on this model.")

def _get_self_attn(layer):
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise RuntimeError("Could not locate attention module on a layer.")

def _find_o_proj(attn, hidden_size: int):
    # common attribute names
    for name in ["o_proj", "out_proj", "wo"]:
        if hasattr(attn, name):
            mod = getattr(attn, name)
            if isinstance(mod, torch.nn.Module):
                return mod, name

    # fallback: scan for a Linear whose out_features == hidden_size
    candidates = []
    for n, m in attn.named_modules():
        if isinstance(m, torch.nn.Linear) and getattr(m, "out_features", None) == hidden_size:
            candidates.append((n, m))
    if not candidates:
        raise RuntimeError("Could not find an output projection module in attention.")

    # prefer name containing 'o' or 'out' if multiple
    candidates.sort(key=lambda x: (("o_proj" not in x[0] and "out" not in x[0] and "wo" not in x[0]), len(x[0])))
    return candidates[0][1], candidates[0][0]

layers = _get_layers(model)
hidden_size = int(getattr(model.config, "hidden_size"))
num_heads = int(getattr(model.config, "num_attention_heads"))
head_dim = hidden_size // num_heads
assert head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_attention_heads"

print("Found layers:", len(layers), "hidden_size:", hidden_size, "num_heads:", num_heads, "head_dim:", head_dim)

# Precompute the per-layer output projection modules so we don't rediscover them every time.
O_PROJS = []
for li, layer in enumerate(layers):
    attn = _get_self_attn(layer)
    mod, mod_name = _find_o_proj(attn, hidden_size)
    O_PROJS.append(mod)
print("o_proj modules discovered:", len(O_PROJS))


# %%
# Cell 26
# ---------------------------
# 2) Head ablation via forward_pre_hook on o_proj
# ---------------------------

class HeadAblator:
    """
    Context manager to ablate specific heads in specific layers.
    spec: Dict[layer_idx -> List[head_idx]]
    """
    def __init__(self, spec: Dict[int, List[int]]):
        self.spec = {int(k): sorted(set(map(int, v))) for k, v in spec.items()}
        self.handles = []

    def _make_pre_hook(self, layer_idx: int, heads_to_zero: List[int]):
        heads = torch.tensor(heads_to_zero, dtype=torch.long)

        def hook(mod, inputs):
            # inputs is a tuple; first item is the o_proj input activations
            x = inputs[0]
            # expected [bs, q_len, hidden], sometimes [bs, hidden] -> treat as q_len=1
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_back = True
            else:
                squeeze_back = False

            bs, q_len, h = x.shape
            if h != hidden_size:
                # can't safely reshape -> no-op
                return (inputs[0],)  # preserve original

            # reshape to [bs, q_len, heads, head_dim]
            xv = x.reshape(bs, q_len, num_heads, head_dim)
            # zero selected heads
            device_heads = heads.to(xv.device)
            xv.index_fill_(dim=2, index=device_heads, value=0.0)
            x2 = xv.reshape(bs, q_len, hidden_size)

            if squeeze_back:
                x2 = x2.squeeze(1)

            # return modified inputs tuple
            return (x2,) + tuple(inputs[1:])

        return hook

    def __enter__(self):
        for layer_idx, heads in self.spec.items():
            if layer_idx < 0 or layer_idx >= len(O_PROJS):
                raise ValueError(f"Layer idx {layer_idx} out of range.")
            if len(heads) == 0:
                continue
            h = O_PROJS[layer_idx].register_forward_pre_hook(self._make_pre_hook(layer_idx, heads))
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        return False



# %%
# Cell 27
# ---------------------------
# 3) One SDPA step (no attentions) that respects cache length
# ---------------------------

@torch.no_grad()
def one_step_sdpa_no_attn(model, token_id: torch.Tensor, past):
    restore = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    past_len = _cache_seq_len(past)
    attn_mask = torch.ones((token_id.shape[0], past_len + 1), device=model.device, dtype=torch.long)
    position_ids = torch.tensor([[past_len]], device=model.device)

    kwargs = dict(
        input_ids=token_id,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        attention_mask=attn_mask,
    )
    try:
        out = model(**kwargs, position_ids=position_ids)
    except TypeError:
        out = model(**kwargs)

    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _set_attn_impl(model, restore or "sdpa")
    return next_tok, out.past_key_values


@torch.no_grad()
def generate_answer_from_prompt(prompt: str, ablation_spec: Optional[Dict[int, List[int]]] = None) -> Dict:
    """
    Returns dict with:
      retrieved (0/1), extracted, gen_txt
    """
    # Prefill WITHOUT ablation (keeps document encoding intact)
    past, input_ids = prefill_build_kv(model, tok, prompt, max_len=cfg4.max_len_encode, chunk_size=cfg4.chunk_size)

    last_tok = input_ids[:, -1:]

    if ablation_spec is None or len(ablation_spec) == 0:
        # last prompt step
        first_tok, past_full = one_step_sdpa_no_attn(model, last_tok, past)
        # generate rest
        gen_txt = greedy_generate_from_past(model, tok, first_tok, past_full, max_new_tokens=cfg4.max_new_tokens)
    else:
        with HeadAblator(ablation_spec):
            first_tok, past_full = one_step_sdpa_no_attn(model, last_tok, past)
            gen_txt = greedy_generate_from_past(model, tok, first_tok, past_full, max_new_tokens=cfg4.max_new_tokens)

    extracted = extract_6digits(gen_txt)
    return {"gen_txt": gen_txt, "extracted": extracted}



# %%
# Cell 28
# ---------------------------
# 4) Dataset builder for reproducible evaluation
# ---------------------------

def build_eval_dataset_for_setting(length: int, needle_frac: float, n: int, seed_base: int = 7777) -> List[Dict]:
    tw = calibrate_total_filler_words_for_setting(
        tok=tok,
        target_tokens=length,
        needle_frac=needle_frac,
        secret=555555,
        seed=0,
        margin_tokens=cfg4.margin_tokens,
        max_tok_cap=cfg4.max_len_encode,
    )

    rng = np.random.default_rng(seed_base + length * 1000 + int(needle_frac * 1000))
    dataset = []
    for i in range(n):
        secret = int(rng.integers(100000, 999999))
        seed = int(seed_base + i + length * 13 + int(needle_frac * 10000))
        prompt, gold = make_prompt_for_setting(
            tok=tok,
            target_tokens=length,
            needle_frac=needle_frac,
            total_filler_words=int(tw),
            secret=secret,
            seed=seed,
            margin_tokens=cfg4.margin_tokens,
            max_tok_cap=cfg4.max_len_encode,
        )
        dataset.append({
            "length": int(length),
            "needle_frac": float(needle_frac),
            "secret": int(secret),
            "gold": str(gold),
            "prompt": prompt,
        })
    return dataset

def build_eval_datasets(cfg4: Day4Config) -> Dict[str, List[Dict]]:
    datasets = {}
    for L in cfg4.lengths:
        for frac in cfg4.needle_fracs:
            key = f"L{L}_p{frac:.2f}"
            datasets[key] = build_eval_dataset_for_setting(L, frac, cfg4.n_prompts_per_setting)
    return datasets

datasets = build_eval_datasets(cfg4)
print({k: len(v) for k, v in datasets.items()})


# %%
# Cell 30.9
# ---------------------------
# Restore Day4-compatible generation helpers
# (Cell 19 overwrote one_step_sdpa_no_attn / greedy helpers with Day3 variants.)
# ---------------------------


@torch.no_grad()
def greedy_generate_from_past(model, tok, first_token: torch.Tensor, past, max_new_tokens: int = 12) -> str:
    """
    Day4/Day5 expects this signature:
      greedy_generate_from_past(model, tok, first_token, past, max_new_tokens) -> str
    """
    # Ensure token dtype is correct
    if first_token.dtype != torch.long:
        first_token = first_token.long()

    toks = [first_token]
    cur = first_token
    for _ in range(max_new_tokens - 1):
        cur, past = one_step_sdpa_no_attn(model, cur, past)
        toks.append(cur)

    gen_ids = torch.cat(toks, dim=1)[0].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True)


# %%
# Cell 29
# ---------------------------
# 5) Evaluation loop
# ---------------------------

@torch.no_grad()
def eval_dataset(dataset: List[Dict], ablation_spec: Optional[Dict[int, List[int]]] = None) -> Dict:
    n = len(dataset)
    ok = 0
    extracted_examples = []
    for ex in dataset:
        out = generate_answer_from_prompt(ex["prompt"], ablation_spec=ablation_spec)
        got = int(out["extracted"] == ex["gold"])
        ok += got
        if len(extracted_examples) < 5:
            extracted_examples.append({
                "gold": ex["gold"],
                "extracted": out["extracted"],
                "gen_txt": out["gen_txt"],
                "ok": got,
            })
    return {
        "n": n,
        "retrieval_rate": ok / max(1, n),
        "examples": extracted_examples,
    }

# Baselines
baseline = {}
for key, ds in datasets.items():
    baseline[key] = eval_dataset(ds, ablation_spec=None)
    print(key, "baseline retrieval_rate =", baseline[key]["retrieval_rate"])


# %%
# Cell 30
# ---------------------------
# 6) Candidate head selection from Day 3 sweep JSON (schema-robust)
# ---------------------------

import os, json, re
import numpy as np
import pandas as pd

DAY3_JSON = os.path.join(ART_DIR, "day3_sweep_results.json")
assert os.path.exists(DAY3_JSON), f"Missing {DAY3_JSON} (run Day 3 first)."
day3 = json.load(open(DAY3_JSON, "r"))

def parse_setting_key(setting: str):
    # supports keys like "L8192_p0.10"
    m = re.match(r"L(\d+)_p([0-9]*\.?[0-9]+)", str(setting))
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))

def day3_long_df_from_json(day3: dict) -> pd.DataFrame:
    rows = []
    for setting, item in day3["settings"].items():
        # Prefer explicit fields if present; otherwise parse from setting string
        L = item.get("target_tokens", item.get("length", None))
        frac = item.get("needle_frac", None)

        if L is None or frac is None:
            L2, frac2 = parse_setting_key(setting)
            if L is None: L = L2
            if frac is None: frac = frac2

        rr = float(item.get("retrieval_rate", np.nan))

        detect = np.array(item["detect_rate"], dtype=float)
        event  = np.array(item["event_rate"], dtype=float)
        score  = np.array(item["retrieval_score"], dtype=float)

        for layer in range(detect.shape[0]):
            for head in range(detect.shape[1]):
                rows.append({
                    "setting": str(setting),
                    "length": int(L) if L is not None else -1,
                    "needle_frac": float(frac) if frac is not None else float("nan"),
                    "retrieval_rate": rr,
                    "layer": int(layer),
                    "head": int(head),
                    "detect_rate": float(detect[layer, head]),
                    "event_rate": float(event[layer, head]),
                    "retrieval_score": float(score[layer, head]),
                })
    return pd.DataFrame(rows)

df3 = day3_long_df_from_json(day3)

def distance_sensitivity(df: pd.DataFrame, far_max: float, near_min: float) -> pd.DataFrame:
    far = df[df["needle_frac"] <= far_max].groupby(["layer","head"], as_index=False).agg(
        far_event=("event_rate","mean"),
        far_detect=("detect_rate","mean"),
        far_score=("retrieval_score","mean"),
    )
    near = df[df["needle_frac"] >= near_min].groupby(["layer","head"], as_index=False).agg(
        near_event=("event_rate","mean"),
        near_detect=("detect_rate","mean"),
        near_score=("retrieval_score","mean"),
    )
    m = far.merge(near, on=["layer","head"], how="inner")
    m["far_minus_near_event"] = m["far_event"] - m["near_event"]
    m["far_minus_near_score"] = m["far_score"] - m["near_score"]
    m["rank_key"] = m["far_event"] + 0.5*m["far_score"]
    return m.sort_values(["rank_key","far_event","far_score"], ascending=False)

df_sens = distance_sensitivity(df3, far_max=cfg4.far_max, near_min=cfg4.near_min)
cand = df_sens.head(cfg4.top_n_heads)[["layer","head","far_event","far_score","near_event","near_score","rank_key"]].copy()

cand_path = os.path.join(ART_DIR, f"{cfg4.out_prefix}_candidate_heads.csv")
cand.to_csv(cand_path, index=False)
print("Saved:", cand_path)
cand.head(15)


# %%
# Cell 30.95
# ---------------------------
# Sanity: ablation changes logits (not just accuracy)
# ---------------------------

import torch
import numpy as np

def one_step_logits_sdpa(model, last_tok, past):
    restore = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    past_len = _cache_seq_len(past)
    pos = torch.tensor([[past_len]], device=model.device, dtype=torch.long)
    attn_mask = torch.ones((1, past_len + 1), device=model.device, dtype=torch.long)

    out = model(
        input_ids=last_tok,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=past,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    logits = out.logits[:, -1, :].detach()

    _set_attn_impl(model, restore or "sdpa")
    return logits

# pick one real prompt from your eval datasets
k = sorted(datasets.keys())[0]
prompt0 = datasets[k][0]["prompt"]

# build cache for prompt
past0, enc0 = prefill_build_kv(model, tok, prompt0, max_len=12000, chunk_size=512)
last_tok0 = tok(prompt0, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)[:, -1:]

# choose a head to test
test_layer, test_head = int(cand.iloc[0]["layer"]), int(cand.iloc[0]["head"])
spec_test = {test_layer: [test_head]}

# logits baseline vs ablated
log0 = one_step_logits_sdpa(model, last_tok0, past0)

with HeadAblator(spec_test):
    past1, _ = prefill_build_kv(model, tok, prompt0, max_len=12000, chunk_size=512)
    log1 = one_step_logits_sdpa(model, last_tok0, past1)

diff = (log1 - log0).abs().max().item()
print(f"Max |Δlogit| for ablation L{test_layer}h{test_head}:", diff)


# %%
# Cell 31
# ---------------------------
# 7) Single-head ablation sweep
# ---------------------------

def _spec_one(layer: int, head: int) -> Dict[int, List[int]]:
    return {int(layer): [int(head)]}

rows = []
for _, r in cand.iterrows():
    layer = int(r["layer"]); head = int(r["head"])
    spec = _spec_one(layer, head)

    for key, ds in datasets.items():
        base_rr = float(baseline[key]["retrieval_rate"])
        ab = eval_dataset(ds, ablation_spec=spec)
        ab_rr = float(ab["retrieval_rate"])
        rows.append({
            "setting": key,
            "layer": layer,
            "head": head,
            "baseline_rr": base_rr,
            "ablated_rr": ab_rr,
            "delta_rr": ab_rr - base_rr,
            "drop_rr": base_rr - ab_rr,
        })
        print(f"{key} ablate (L{layer},h{head})  base={base_rr:.3f}  ablated={ab_rr:.3f}  drop={base_rr-ab_rr:.3f}")

df_ab = pd.DataFrame(rows)
ab_csv = os.path.join(ART_DIR, f"{cfg4.out_prefix}_single_head_ablations.csv")
df_ab.to_csv(ab_csv, index=False)
print("Saved:", ab_csv)

# Show biggest drops on FAR setting (usually the most interesting)
far_key = f"L{cfg4.lengths[0]}_p{cfg4.needle_fracs[0]:.2f}"
df_ab[df_ab["setting"] == far_key].sort_values("drop_rr", ascending=False).head(20)


# %%
# Cell 32
# ---------------------------
# 8) Group ablations (top-K) + layer-matched DISJOINT random controls
#    (report-grade: multiple random draws + summary stats)
# ---------------------------

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Iterable

def spec_from_pairs(pairs: Iterable[Tuple[int,int]]) -> Dict[int, List[int]]:
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec

def layer_counts(pairs: List[Tuple[int,int]]) -> Dict[int,int]:
    c = {}
    for L, _H in pairs:
        c[int(L)] = c.get(int(L), 0) + 1
    return c

def sample_layer_matched_disjoint(top_pairs: List[Tuple[int,int]], seed: int, banned: set) -> List[Tuple[int,int]]:
    """
    Sample random pairs with the SAME per-layer head counts as top_pairs,
    and ensure none overlap with banned (usually top_pairs).
    """
    rng = np.random.default_rng(seed)
    counts = layer_counts(top_pairs)

    out = []
    for L, k in sorted(counts.items()):
        candidates = [(L, h) for h in range(n_heads) if (L, h) not in banned]
        if len(candidates) < k:
            raise RuntimeError(f"Not enough candidates in layer {L}: need {k}, have {len(candidates)}")
        idx = rng.choice(len(candidates), size=k, replace=False)
        for j in idx:
            out.append(candidates[int(j)])
    return out

# choose K (keep your original default, but you can also try K=8 since you saw ~7–8 “strong” heads)
topK = min(10, len(cand))
# topK = 8 
top_pairs = [(int(cand.iloc[i]["layer"]), int(cand.iloc[i]["head"])) for i in range(topK)]
spec_top = spec_from_pairs(top_pairs)

# multiple random controls for a tighter story
N_RAND = 20
rand_seeds = [4242 + i for i in range(N_RAND)]
banned = set(top_pairs)

rand_pairs_list = [sample_layer_matched_disjoint(top_pairs, seed=s, banned=banned) for s in rand_seeds]
rand_specs_list = [spec_from_pairs(p) for p in rand_pairs_list]

print("topK:", topK)
print("top_pairs:", top_pairs)
print("example rand_pairs[0]:", rand_pairs_list[0])

group_rows = []

for key, ds in datasets.items():
    base_rr = float(baseline[key]["retrieval_rate"])

    rr_top = float(eval_dataset(ds, ablation_spec=spec_top)["retrieval_rate"])
    drop_top = base_rr - rr_top

    rr_rands = []
    for spec_r in rand_specs_list:
        rr_rands.append(float(eval_dataset(ds, ablation_spec=spec_r)["retrieval_rate"]))
    rr_rands = np.array(rr_rands, dtype=float)

    drop_rands = base_rr - rr_rands
    group_rows.append({
        "setting": key,
        "baseline_rr": base_rr,
        "topK": topK,
        "ablated_topK_rr": rr_top,
        "drop_topK_rr": drop_top,
        "randK_mean_rr": float(rr_rands.mean()),
        "randK_std_rr": float(rr_rands.std()),
        "randK_mean_drop": float(drop_rands.mean()),
        "randK_std_drop": float(drop_rands.std()),
        "top_pairs": top_pairs,
        "rand_seeds": rand_seeds,
    })

    print(f"{key}: drop_topK={drop_top:.4f} | randK mean drop={drop_rands.mean():.4f} ± {drop_rands.std():.4f}")

df_group = pd.DataFrame(group_rows)
group_csv = os.path.join(ART_DIR, f"{cfg4.out_prefix}_group_ablations.csv")
df_group.to_csv(group_csv, index=False)
print("Saved:", group_csv)
df_group


# %%
# Cell 33
# ---------------------------
# 9) Plot biggest causal heads (far setting)
# ---------------------------

far_key = f"L{cfg4.lengths[0]}_p{cfg4.needle_fracs[0]:.2f}"
df_far = df_ab[df_ab["setting"] == far_key].copy()
df_far["head_id"] = df_far.apply(lambda r: f"L{int(r.layer)}h{int(r.head)}", axis=1)
df_far = df_far.sort_values("drop_rr", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df_far["head_id"], df_far["drop_rr"])
ax.set_title(f"Top-20 single-head ablation drops — {far_key}")
ax.set_ylabel("retrieval rate drop")
ax.set_xlabel("head")
ax.tick_params(axis="x", rotation=90)

png = os.path.join(ART_DIR, f"{cfg4.out_prefix}_top_ablation_drops_{far_key}.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", png)

df_far[["layer","head","baseline_rr","ablated_rr","drop_rr"]]


# %%
# Cell 34 (updated)
import numpy as np
import re

def prompt_len_and_distance(prompt: str, gold: str):
    # IMPORTANT: match the rest of the notebook: add_special_tokens=False
    enc = tok(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # locate the last marker block in raw string
    ms = prompt.rfind(NEEDLE_START)
    if ms == -1:
        raise RuntimeError("NEEDLE_START not found in prompt text")
    me = prompt.find(NEEDLE_END, ms + len(NEEDLE_START))
    if me == -1 or me <= ms:
        raise RuntimeError("NEEDLE_END not found after NEEDLE_START in prompt text")

    region_start = ms + len(NEEDLE_START)
    region_end = me

    # locate the gold digits inside marker region
    a0 = prompt.find(str(gold), region_start, region_end)
    if a0 == -1:
        # fallback: regex search in region
        region_txt = prompt[region_start:region_end]
        m = re.search(re.escape(str(gold)), region_txt)
        if not m:
            raise RuntimeError("Gold answer not found inside NEEDLE region")
        a0 = region_start + m.start()
    a1 = a0 + len(str(gold))

    # map char span -> token span using offsets
    tok_start = None
    tok_end = None
    for i, (s, e) in enumerate(offsets):
        if e > a0 and s < a1:   # overlap
            if tok_start is None:
                tok_start = i
            tok_end = i + 1

    if tok_start is None or tok_end is None:
        raise RuntimeError("Could not map gold char span to token span")

    # distance from final prompt token (query position) to end of answer span
    dist = (len(ids) - 1) - tok_end
    return len(ids), int(dist)

for k, ds in datasets.items():
    lens, dists = [], []
    for ex in ds:
        L, D = prompt_len_and_distance(ex["prompt"], ex["gold"])
        lens.append(L); dists.append(D)
    lens = np.array(lens); dists = np.array(dists)
    print(
        f"{k}: prompt_len mean={lens.mean():.1f} (min={lens.min()}, max={lens.max()}), "
        f"distance mean={dists.mean():.1f} (min={dists.min()}, max={dists.max()})"
    )


# %%
# Cell 35
def spec_all_heads_all_layers():
    return {li: list(range(num_heads)) for li in range(len(layers))}

def quick_check(ds_key, k=10, spec=None):
    ds = datasets[ds_key]
    ok = 0
    for ex in ds[:k]:
        out = generate_answer_from_prompt(ex["prompt"], ablation_spec=spec)
        got = (out["extracted"] == ex["gold"])
        ok += int(got)
        print("gold=", ex["gold"], " extracted=", out["extracted"], " ok=", got, " gen=", repr(out["gen_txt"][:40]))
    print("OK:", ok, "/", k)

far_key  = "L8192_p0.10"
near_key = "L8192_p0.90"

print("Baseline FAR")
quick_check(far_key, k=5, spec=None)

print("\nABLATE ALL HEADS ALL LAYERS FAR (should tank)")
quick_check(far_key, k=5, spec=spec_all_heads_all_layers())


# %%
# Cell 36
# ==========================================
# Day 4b — log-probability ablations
# ==========================================

@torch.no_grad()
def one_step_sdpa_logits(model, token_id: torch.Tensor, past):
    """
    One SDPA step that returns logits and updated past.
    No attentions; identical cache semantics to one_step_sdpa_no_attn.
    """
    restore = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    past_len = _cache_seq_len(past)
    attn_mask = torch.ones((token_id.shape[0], past_len + 1),
                           device=model.device, dtype=torch.long)
    position_ids = torch.tensor([[past_len]], device=model.device)

    kwargs = dict(
        input_ids=token_id,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        attention_mask=attn_mask,
    )
    try:
        out = model(**kwargs, position_ids=position_ids)
    except TypeError:
        out = model(**kwargs)

    logits = out.logits[:, -1, :]

    _set_attn_impl(model, restore or "sdpa")
    return logits, out.past_key_values



# %%
# Cell 37
@torch.no_grad()
def gold_logprob_for_example(prompt: str, gold: str,
                             ablation_spec: Optional[Dict[int, List[int]]] = None) -> float:
    """
    Average log-probability per digit for the *exact* gold digit string,
    under greedy teacher-forced decoding:
      p(gold[0]), p(gold[1] | gold[0]), ...
    If ablation_spec is provided, apply that HeadAblator during the forward pass.
    """
    gold_ids = tok(gold, add_special_tokens=False).input_ids
    if len(gold_ids) == 0:
        return float("nan")

    # Same prefill as Day 3/4
    past, input_ids = prefill_build_kv(
        model, tok, prompt,
        max_len=cfg4.max_len_encode,
        chunk_size=cfg4.chunk_size,
    )
    last_tok = input_ids[:, -1:]

    def run():
        lp_total = 0.0
        logits, cur_past = one_step_sdpa_logits(model, last_tok, past)
        for tid in gold_ids:
            log_probs = torch.log_softmax(logits, dim=-1)
            lp_total += float(log_probs[0, tid].detach().cpu())
            nxt = torch.tensor([[tid]], device=model.device)
            logits, cur_past = one_step_sdpa_logits(model, nxt, cur_past)
        # mean log-prob per digit, easier to compare across examples
        return lp_total / len(gold_ids)

    if ablation_spec and len(ablation_spec) > 0:
        with HeadAblator(ablation_spec):
            return run()
    else:
        return run()



# %%
# Cell 38
@torch.no_grad()
def eval_logprob_dataset(ds: List[Dict],
                         ablation_spec: Optional[Dict[int, List[int]]] = None) -> Dict:
    lps = []
    for ex in ds:
        lp = gold_logprob_for_example(ex["prompt"], ex["gold"], ablation_spec=ablation_spec)
        lps.append(lp)
    arr = np.array(lps, dtype=np.float64)
    return {
        "n": len(ds),
        "mean_logprob": float(arr.mean()),
        "std_logprob": float(arr.std()),
    }

# Quick sanity: far vs near, no ablation
far_key  = "L8192_p0.10"
near_key = "L8192_p0.90"

logp_baseline = {
    far_key:  eval_logprob_dataset(datasets[far_key],  None),
    near_key: eval_logprob_dataset(datasets[near_key], None),
}
print("Baseline mean log-probs:")
for k, v in logp_baseline.items():
    print(k, "mean=", v["mean_logprob"], "std=", v["std_logprob"])


# %%
# Cell 39
# ---------------------------
# Single-head log-prob ablations
# ---------------------------

logp_rows = []

for _, r in cand.iterrows():
    layer = int(r["layer"]); head = int(r["head"])
    spec = {layer: [head]}

    for key, ds in datasets.items():
        base = eval_logprob_dataset(ds, ablation_spec=None)
        ab   = eval_logprob_dataset(ds, ablation_spec=spec)

        logp_rows.append({
            "setting": key,
            "layer": layer,
            "head": head,
            "baseline_mean_logprob": base["mean_logprob"],
            "ablated_mean_logprob": ab["mean_logprob"],
            "delta_logprob": ab["mean_logprob"] - base["mean_logprob"],   # <= 0 if ablation hurts
        })
        print(f"{key} L{layer}h{head}  base={base['mean_logprob']:.3f}  "
              f"ablated={ab['mean_logprob']:.3f}  delta={ab['mean_logprob']-base['mean_logprob']:.3f}")

df_logp_single = pd.DataFrame(logp_rows)
logp_single_csv = os.path.join(ART_DIR, f"{cfg4.out_prefix}_single_head_logprob.csv")
df_logp_single.to_csv(logp_single_csv, index=False)
print("Saved:", logp_single_csv)

# Focus on FAR setting for ranking
df_logp_far = df_logp_single[df_logp_single["setting"] == far_key].copy()
df_logp_far = df_logp_far.sort_values("delta_logprob")  # most negative = biggest damage
df_logp_far.head(20)


# %%
# Cell 40
# ---------------------------
# Group log-prob ablations (topK vs layer-matched disjoint randomK)
# ---------------------------

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Iterable

def spec_from_pairs(pairs):
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec

def layer_counts(pairs):
    c = {}
    for L, _H in pairs:
        c[int(L)] = c.get(int(L), 0) + 1
    return c

def sample_layer_matched_disjoint(top_pairs, seed: int, banned: set):
    rng = np.random.default_rng(seed)
    counts = layer_counts(top_pairs)
    out = []
    for L, k in sorted(counts.items()):
        candidates = [(L, h) for h in range(n_heads) if (L, h) not in banned]
        if len(candidates) < k:
            raise RuntimeError(f"Not enough candidates in layer {L}: need {k}, have {len(candidates)}")
        idx = rng.choice(len(candidates), size=k, replace=False)
        for j in idx:
            out.append(candidates[int(j)])
    return out

topK = min(10, len(cand))
top_pairs = [(int(cand.iloc[i]["layer"]), int(cand.iloc[i]["head"])) for i in range(topK)]
spec_top  = spec_from_pairs(top_pairs)

N_RAND = 20 # make it 20
rand_seeds = [4242 + i for i in range(N_RAND)]
banned = set(top_pairs)
rand_pairs_list = [sample_layer_matched_disjoint(top_pairs, seed=s, banned=banned) for s in rand_seeds]
rand_specs_list = [spec_from_pairs(p) for p in rand_pairs_list]

group_logp_rows = []
for key, ds in datasets.items():
    base = eval_logprob_dataset(ds, None)
    top  = eval_logprob_dataset(ds, spec_top)

    rand_vals = []
    for spec_r in rand_specs_list:
        rnd = eval_logprob_dataset(ds, spec_r)
        rand_vals.append(rnd["mean_logprob"])
    rand_vals = np.array(rand_vals, dtype=float)

    group_logp_rows.append({
        "setting": key,
        "topK": topK,
        "baseline_mean_logprob": base["mean_logprob"],
        "topK_mean_logprob": top["mean_logprob"],
        "delta_topK": top["mean_logprob"] - base["mean_logprob"],
        "randK_mean_logprob": float(rand_vals.mean()),
        "randK_std_logprob": float(rand_vals.std()),
        "delta_randK_mean": float(rand_vals.mean() - base["mean_logprob"]),
        "delta_randK_std": float(rand_vals.std()),
        "top_pairs": top_pairs,
        "rand_seeds": rand_seeds,
    })

    print(f"{key}: Δlogp_topK={top['mean_logprob']-base['mean_logprob']:.6f} | "
          f"Δlogp_randK(mean)={rand_vals.mean()-base['mean_logprob']:.6f} ± {rand_vals.std():.6f}")

df_group_logp = pd.DataFrame(group_logp_rows)
out_csv = os.path.join(ART_DIR, f"{cfg4.out_prefix}_group_logprob.csv")
df_group_logp.to_csv(out_csv, index=False)
print("Saved:", out_csv)
df_group_logp


# %%
# Cell 41
# ============================================================
# Day 5 — Activation patching (sufficiency) for retrieval heads
# ============================================================

import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager

model.eval()

# ---------------------------
# Helpers: attention impl context + one-step logits
# ---------------------------

@contextmanager
def use_attn_impl(model, mode: str):
    restore = _get_attn_impl(model)
    _set_attn_impl(model, mode)
    try:
        yield
    finally:
        _set_attn_impl(model, restore or "sdpa")

@torch.no_grad()
def one_step_logits(token_id: torch.Tensor, past):
    """
    One forward step that returns logits and updated past.
    Does NOT toggle attention impl; wrap caller in use_attn_impl().
    """
    past_len = _cache_seq_len(past)
    step_len = token_id.shape[1]

    attn_mask = torch.ones((token_id.shape[0], past_len + step_len),
                           device=model.device, dtype=torch.long)
    position_ids = torch.arange(past_len, past_len + step_len,
                                device=model.device).unsqueeze(0)

    kwargs = dict(
        input_ids=token_id,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        attention_mask=attn_mask,
    )
    try:
        out = model(**kwargs, position_ids=position_ids)
    except TypeError:
        out = model(**kwargs)

    logits = out.logits[:, -1, :]
    return logits, out.past_key_values

# ---------------------------
# Secret pairing: keep prompt lengths matched (tokenization-aware)
# ---------------------------

def secret_doc_toklen(tok, secret: int) -> int:
    # secret appears as "... is {secret}.", so include leading space + period
    s = f" {secret}."
    return len(tok(s, add_special_tokens=False).input_ids)

def sample_secret_pair_same_toklen(tok, rng, max_tries: int = 20000):
    s1 = int(rng.integers(100000, 999999))
    L = secret_doc_toklen(tok, s1)
    for _ in range(max_tries):
        s2 = int(rng.integers(100000, 999999))
        if s2 != s1 and secret_doc_toklen(tok, s2) == L:
            return s1, s2
    raise RuntimeError("Failed to sample token-length matched secrets (increase max_tries).")

# ---------------------------
# Dataset: paired clean/corrupt prompts with identical filler
# ---------------------------

@dataclass
class Day5Config:
    length: int = 8192
    needle_frac: float = 0.10
    n_examples: int = 40
    seed_base: int = 9090

    chunk_size: int = 1024
    max_len_encode: int = 16384
    margin_tokens: int = 96
    max_new_tokens: int = 14

    topK: int = 10
    random_seed: int = 4242

cfg5 = Day5Config()

def build_paired_dataset(cfg5: Day5Config):
    tw = calibrate_total_filler_words_for_setting(
        tok=tok,
        target_tokens=cfg5.length,
        needle_frac=cfg5.needle_frac,
        secret=555555,
        seed=0,
        margin_tokens=cfg5.margin_tokens,
        max_tok_cap=cfg5.max_len_encode,
    )

    rng = np.random.default_rng(cfg5.seed_base + cfg5.length * 17 + int(cfg5.needle_frac * 1000))
    data = []
    for i in range(cfg5.n_examples):
        s_clean, s_corrupt = sample_secret_pair_same_toklen(tok, rng)
        seed = int(cfg5.seed_base + i + cfg5.length * 13 + int(cfg5.needle_frac * 10000))

        prompt_clean, gold_clean = make_prompt_for_setting(
            tok=tok,
            target_tokens=cfg5.length,
            needle_frac=cfg5.needle_frac,
            total_filler_words=int(tw),
            secret=int(s_clean),
            seed=seed,
            margin_tokens=cfg5.margin_tokens,
            max_tok_cap=cfg5.max_len_encode,
        )
        prompt_corrupt, gold_corrupt = make_prompt_for_setting(
            tok=tok,
            target_tokens=cfg5.length,
            needle_frac=cfg5.needle_frac,
            total_filler_words=int(tw),
            secret=int(s_corrupt),
            seed=seed,  # SAME SEED => SAME filler words
            margin_tokens=cfg5.margin_tokens,
            max_tok_cap=cfg5.max_len_encode,
        )

        data.append({
            "i": i,
            "seed": seed,
            "gold_clean": gold_clean,
            "gold_corrupt": gold_corrupt,
            "prompt_clean": prompt_clean,
            "prompt_corrupt": prompt_corrupt,
        })
    return data

paired = build_paired_dataset(cfg5)
print("paired examples:", len(paired))
print("example golds:", paired[0]["gold_clean"], paired[0]["gold_corrupt"])


# %%
# Cell 42
# ---------------------------
# Capture + patch at o_proj input (head-separated)
# ---------------------------

class OProjInputCacher:
    """Capture the input to each layer's o_proj during a forward pass."""
    def __init__(self):
        self.cache = {}
        self.handles = []

    def _hook(self, layer_idx: int):
        def hook(mod, inputs):
            x = inputs[0]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            # clone so later in-place ops can't mutate our snapshot
            self.cache[layer_idx] = x.detach().clone()
            return None
        return hook

    def __enter__(self):
        for li, mod in enumerate(O_PROJS):
            self.handles.append(mod.register_forward_pre_hook(self._hook(li)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles = []
        return False


class HeadPatcher:
    """
    Patch selected heads' o_proj-input slices from clean_cache into the current forward pass.
    spec: Dict[layer_idx -> List[head_idx]]
    """
    def __init__(self, clean_cache: dict, spec: dict):
        self.clean_cache = clean_cache
        self.spec = {int(k): sorted(set(map(int, v))) for k, v in spec.items()}
        self.handles = []

    def _hook(self, layer_idx: int, heads_to_patch: list):
        heads = torch.tensor(heads_to_patch, dtype=torch.long)

        def hook(mod, inputs):
            x = inputs[0]
            squeeze_back = False
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_back = True

            x = x.contiguous()
            if layer_idx not in self.clean_cache:
                return None

            x_clean = self.clean_cache[layer_idx].to(x.device)
            if x_clean.dim() == 2:
                x_clean = x_clean.unsqueeze(1)
            x_clean = x_clean.contiguous()

            if x.shape[-1] != hidden_size or x_clean.shape[-1] != hidden_size:
                return None

            bs, q_len, _ = x.shape
            xv = x.view(bs, q_len, num_heads, head_dim)
            xc = x_clean.view(bs, q_len, num_heads, head_dim)

            idx = heads.to(x.device)
            # replace selected heads in-place
            xv.index_copy_(2, idx, xc.index_select(2, idx))

            x2 = xv.view(bs, q_len, hidden_size)
            if squeeze_back:
                x2 = x2.squeeze(1)

            return (x2,) + tuple(inputs[1:])

        return hook

    def __enter__(self):
        for li, heads in self.spec.items():
            if not heads:
                continue
            self.handles.append(O_PROJS[li].register_forward_pre_hook(self._hook(li, heads)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles = []
        return False



# %%
# Cell 43
# ---------------------------
# Scoring: mean teacher-forced logprob for the "clean" gold on the corrupt prompt
# Patch is applied ONLY for the query step (last prompt token); effect persists via past.
# ---------------------------

@torch.no_grad()
def prefill_last_tok(prompt: str, max_len: int, chunk_size: int):
    past, input_ids = prefill_build_kv(model, tok, prompt, max_len=max_len, chunk_size=chunk_size)
    last_tok = input_ids[:, -1:]
    return past, last_tok

@torch.no_grad()
def mean_logprob_from_prefill(past, last_tok, gold: str, patch_cache=None, patch_spec=None) -> float:
    gold_ids = tok(gold, add_special_tokens=False).input_ids
    if len(gold_ids) == 0:
        return float("nan")

    with use_attn_impl(model, "sdpa"):
        if patch_cache is None or patch_spec is None or len(patch_spec) == 0:
            logits, cur_past = one_step_logits(last_tok, past)
        else:
            with HeadPatcher(patch_cache, patch_spec):
                logits, cur_past = one_step_logits(last_tok, past)

        lp = 0.0
        for tid in gold_ids:
            lp += float(torch.log_softmax(logits, dim=-1)[0, tid].detach().cpu())
            nxt = torch.tensor([[tid]], device=model.device)
            logits, cur_past = one_step_logits(nxt, cur_past)

    return lp / len(gold_ids)

@torch.no_grad()
def capture_clean_cache(prompt_clean: str):
    past, last_tok = prefill_last_tok(prompt_clean, cfg5.max_len_encode, cfg5.chunk_size)
    with use_attn_impl(model, "sdpa"):
        with OProjInputCacher() as cacher:
            _logits, _past = one_step_logits(last_tok, past)
    return cacher.cache

def spec_from_pairs(pairs):
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec



# %%
# Cell 44
# ---------------------------
# Choose which heads to patch (prefer logprob ranking) + layer-matched DISJOINT random control
# ---------------------------

import os
import numpy as np
import pandas as pd

CAND_PATH = os.path.join(ART_DIR, "day4_candidate_heads.csv")
assert os.path.exists(CAND_PATH), f"Missing {CAND_PATH} (run Cell 30 first)."
cand = pd.read_csv(CAND_PATH)

FAR_KEY = f"L{cfg5.length}_p{cfg5.needle_frac:.2f}"
LOGP_PATH = os.path.join(ART_DIR, "day4_single_head_logprob.csv")

def spec_from_pairs(pairs):
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec

def layer_counts(pairs):
    c = {}
    for L, _H in pairs:
        c[int(L)] = c.get(int(L), 0) + 1
    return c

def sample_layer_matched_disjoint(top_pairs, seed: int, banned: set):
    rng = np.random.default_rng(seed)
    counts = layer_counts(top_pairs)
    out = []
    for L, k in sorted(counts.items()):
        candidates = [(L, h) for h in range(n_heads) if (L, h) not in banned]
        if len(candidates) < k:
            raise RuntimeError(f"Not enough candidates in layer {L}: need {k}, have {len(candidates)}")
        idx = rng.choice(len(candidates), size=k, replace=False)
        for j in idx:
            out.append(candidates[int(j)])
    return out

if os.path.exists(LOGP_PATH):
    df_logp = pd.read_csv(LOGP_PATH)
    df_logp_far = df_logp[df_logp["setting"] == FAR_KEY].sort_values("delta_logprob")  # most negative = most important
    ranked_pairs = list(zip(df_logp_far["layer"].astype(int), df_logp_far["head"].astype(int)))
    print("Using logprob ablation ranking:", LOGP_PATH)
else:
    ranked_pairs = list(zip(cand["layer"].astype(int), cand["head"].astype(int)))
    print("Using Day4 candidate ranking:", CAND_PATH)

top_pairs = ranked_pairs[:cfg5.topK]
spec_top = spec_from_pairs(top_pairs)

# layer-matched, disjoint randomK
banned = set(top_pairs)
rand_pairs = sample_layer_matched_disjoint(top_pairs, seed=cfg5.random_seed, banned=banned)
spec_rand = spec_from_pairs(rand_pairs)

print("top_pairs:", top_pairs)
print("rand_pairs:", rand_pairs)
print("per-layer counts top:", layer_counts(top_pairs))
print("per-layer counts rnd:", layer_counts(rand_pairs))


# %%
# Cell 45
# ---------------------------
# Run patching evaluation (group patch: topK vs randomK)
# ---------------------------

rows = []
for ex in tqdm(paired, desc="Day5 patch eval"):
    # prefill corrupt ONCE
    past_corrupt, last_tok_corrupt = prefill_last_tok(ex["prompt_corrupt"], cfg5.max_len_encode, cfg5.chunk_size)

    # baseline: logprob of clean answer under corrupt prompt
    lp_base = mean_logprob_from_prefill(
        past_corrupt, last_tok_corrupt,
        gold=ex["gold_clean"],
        patch_cache=None,
        patch_spec=None
    )

    # capture clean cache (o_proj inputs at query step)
    clean_cache = capture_clean_cache(ex["prompt_clean"])

    # patch topK
    lp_top = mean_logprob_from_prefill(
        past_corrupt, last_tok_corrupt,
        gold=ex["gold_clean"],
        patch_cache=clean_cache,
        patch_spec=spec_top
    )

    # patch randomK control
    lp_rnd = mean_logprob_from_prefill(
        past_corrupt, last_tok_corrupt,
        gold=ex["gold_clean"],
        patch_cache=clean_cache,
        patch_spec=spec_rand
    )

    rows.append({
        "i": ex["i"],
        "gold_clean": ex["gold_clean"],
        "gold_corrupt": ex["gold_corrupt"],
        "lp_base": lp_base,
        "lp_patch_topK": lp_top,
        "lp_patch_randK": lp_rnd,
        "delta_topK": lp_top - lp_base,
        "delta_randK": lp_rnd - lp_base,
    })

df_patch = pd.DataFrame(rows)
out_csv = os.path.join(ART_DIR, "day5_patch_group_results.csv")
df_patch.to_csv(out_csv, index=False)
print("Saved:", out_csv)

print(df_patch[["delta_topK","delta_randK"]].describe())


# %%
# # Cell 46
# # ---------------------------
# # LongBench v2 micro-slice external validation (≤10 items)
# # Baseline vs TopK vs layer-matched disjoint RandomK
# # ---------------------------

# import os, re, json
# from collections import Counter
# from contextlib import nullcontext

# import numpy as np
# import pandas as pd
# import torch

# # ---- requirements from earlier cells ----
# assert "HeadAblator" in globals(), "Missing HeadAblator (run Day4 setup cells first)."
# assert "prefill_build_kv" in globals(), "Missing prefill_build_kv (run Cell 19 first)."
# assert "one_step_sdpa_no_attn" in globals(), "Missing one_step_sdpa_no_attn (run Cell 30.9 restore cell)."
# assert "greedy_generate_from_past" in globals(), "Missing greedy_generate_from_past (run Cell 30.9 restore cell)."
# assert "tok" in globals() and "model" in globals(), "Missing tok/model."

# # Use your existing cap (from Day4 config if present)
# MAX_ENCODE = int(getattr(globals().get("cfg4", object()), "max_len_encode", 12000))
# MAX_NEW_TOKENS = 6
# N_ITEMS = 10
# N_RAND = 5
# TOPK = 8

# # ---- load LongBench v2 (streaming to avoid pulling everything eagerly) ----
# try:
#     from datasets import load_dataset
# except Exception as e:
#     raise RuntimeError("Please `pip install datasets` (or ensure it's installed) before running this cell.") from e

# def _try_load_longbench_v2_stream():
#     # HF card suggests THUDM/LongBench-v2; some mirrors exist (e.g., zai-org/LongBench-v2).
#     # We'll try both.
#     last_err = None
#     for name in ["THUDM/LongBench-v2", "zai-org/LongBench-v2"]:
#         try:
#             ds = load_dataset(name, split="train", streaming=True)
#             print("Loaded streaming dataset:", name)
#             return ds, name
#         except Exception as e:
#             last_err = e
#     raise RuntimeError(f"Could not load LongBench v2 from HF (tried THUDM/LongBench-v2 and zai-org/LongBench-v2). Last error: {last_err}")

# lb_stream, lb_source = _try_load_longbench_v2_stream()

# # ---- prompt building + truncation helpers ----
# _LET_RE = re.compile(r"\b([ABCD])\b")

# def build_mcq_prompt(context: str, question: str, A: str, B: str, C: str, D: str) -> str:
#     # Keep it strict to force letter-only output
#     return (
#         "You are given a long context and a multiple-choice question.\n"
#         "Choose the correct option.\n"
#         "Answer with ONLY a single letter: A, B, C, or D.\n\n"
#         "=== CONTEXT START ===\n"
#         f"{context}\n"
#         "=== CONTEXT END ===\n\n"
#         f"Question: {question}\n"
#         f"A. {A}\n"
#         f"B. {B}\n"
#         f"C. {C}\n"
#         f"D. {D}\n\n"
#         "Final answer:"
#     )

# def trim_context_to_token_budget(context: str, base_prompt_no_context: str, max_tokens: int) -> (str, bool, int):
#     """
#     Returns (trimmed_context, truncated_flag, approx_context_tokens_used)
#     Uses a head+tail token trim if needed.
#     """
#     # Pre-trim huge contexts at char level to avoid pathological tokenization cost
#     truncated = False
#     if context is None:
#         context = ""
#     if len(context) > 50000:
#         truncated = True
#         half = 25000
#         context = context[:half] + "\n...\n" + context[-half:]

#     base_ids = tok(base_prompt_no_context, add_special_tokens=False).input_ids
#     budget_for_context = max_tokens - len(base_ids) - 16  # small safety margin
#     if budget_for_context <= 32:
#         # can't fit much; return minimal
#         truncated = True
#         return context[:1024], True, 0

#     ctx_ids = tok(context, add_special_tokens=False).input_ids
#     if len(ctx_ids) <= budget_for_context:
#         return context, truncated, len(ctx_ids)

#     # Need token trim: keep head + tail
#     truncated = True
#     keep = int(budget_for_context)
#     keep_head = keep // 2
#     keep_tail = keep - keep_head
#     head_ids = ctx_ids[:keep_head]
#     tail_ids = ctx_ids[-keep_tail:] if keep_tail > 0 else []
#     trimmed = tok.decode(head_ids) + "\n...\n" + tok.decode(tail_ids)
#     return trimmed, True, keep

# def prompt_token_len(prompt: str) -> int:
#     return len(tok(prompt, add_special_tokens=False).input_ids)

# # ---- build micro-slice examples that fit token budget ----
# examples = []
# for ex in lb_stream:
#     # LongBench v2 standardized fields per HF card
#     # context, question, choice_A/B/C/D, answer (A/B/C/D)
#     context = ex.get("context", "")
#     question = ex.get("question", "")
#     A = ex.get("choice_A", "")
#     B = ex.get("choice_B", "")
#     C = ex.get("choice_C", "")
#     D = ex.get("choice_D", "")
#     gold = ex.get("answer", "")

#     # Build base prompt with placeholder context just to compute base length
#     base_prompt_no_context = build_mcq_prompt("", question, A, B, C, D)

#     ctx_trimmed, was_trunc, ctx_tok_used = trim_context_to_token_budget(
#         context=context,
#         base_prompt_no_context=base_prompt_no_context,
#         max_tokens=MAX_ENCODE,
#     )
#     prompt = build_mcq_prompt(ctx_trimmed, question, A, B, C, D)
#     L = prompt_token_len(prompt)

#     # Keep only prompts that fit the encode cap
#     if L <= MAX_ENCODE and gold in ["A", "B", "C", "D"]:
#         examples.append({
#             "id": ex.get("_id", None),
#             "domain": ex.get("domain", None),
#             "sub_domain": ex.get("sub_domain", None),
#             "difficulty": ex.get("difficulty", None),
#             "length": ex.get("length", None),
#             "prompt": prompt,
#             "gold": gold,
#             "prompt_len_tokens": int(L),
#             "context_tokens_used": int(ctx_tok_used),
#             "truncated": int(was_trunc),
#         })
#         if len(examples) >= N_ITEMS:
#             break

# if len(examples) == 0:
#     raise RuntimeError(f"Could not collect any LongBench-v2 examples under MAX_ENCODE={MAX_ENCODE}. Increase cap or reduce N_ITEMS.")

# print(f"Collected {len(examples)} LongBench-v2 examples from {lb_source}. "
#       f"Truncated: {sum(e['truncated'] for e in examples)}/{len(examples)}")

# # ---- build TopK spec from your detection candidates ----
# def spec_from_pairs(pairs):
#     spec = {}
#     for L, H in pairs:
#         spec.setdefault(int(L), []).append(int(H))
#     for L in list(spec.keys()):
#         spec[L] = sorted(set(spec[L]))
#     return spec

# # Reuse top_pairs if already computed; else derive from cand
# if "top_pairs" in globals() and isinstance(globals()["top_pairs"], (list, tuple)) and len(globals()["top_pairs"]) > 0:
#     _top_pairs = [(int(L), int(H)) for (L, H) in globals()["top_pairs"][:TOPK]]
# else:
#     assert "cand" in globals(), "Need cand DataFrame from Cell 30 (candidate heads)."
#     _top_pairs = [(int(cand.iloc[i]["layer"]), int(cand.iloc[i]["head"])) for i in range(min(TOPK, len(cand)))]

# # ---- layer-matched, disjoint random sampler ----
# def sample_layer_matched_disjoint_random(top_pairs, seed=0):
#     rng = np.random.default_rng(seed)
#     top_set = set((int(L), int(H)) for (L, H) in top_pairs)
#     layer_counts = Counter([int(L) for (L, _) in top_pairs])

#     out = []
#     for L, k in layer_counts.items():
#         # sample heads within same layer, excluding TopK heads in that layer
#         banned = set(H for (LL, H) in top_set if LL == L)
#         choices = [h for h in range(n_heads) if h not in banned]
#         if len(choices) < k:
#             # fall back: allow overlap within layer only if unavoidable (rare)
#             choices = [h for h in range(n_heads)]
#         picked = rng.choice(choices, size=k, replace=False).tolist()
#         out.extend([(L, int(h)) for h in picked])

#     # ensure global disjointness if possible
#     out = [(L, H) for (L, H) in out if (L, H) not in top_set]
#     # If we lost some due to disjointness filtering, refill anywhere disjoint
#     while len(out) < len(top_pairs):
#         L = int(rng.integers(0, n_layers))
#         H = int(rng.integers(0, n_heads))
#         if (L, H) in top_set or (L, H) in out:
#             continue
#         out.append((L, H))
#     return out[:len(top_pairs)]

# spec_top = spec_from_pairs(_top_pairs)

# # ---- MCQ generation + evaluation ----
# @torch.no_grad()
# def generate_mcq_letter(prompt: str, ablation_spec=None, max_new_tokens: int = 6):
#     # build KV cache for prompt
#     past, _ = prefill_build_kv(model, tok, prompt, max_len=MAX_ENCODE, chunk_size=512)
#     enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
#     last_tok = enc["input_ids"].to(model.device)[:, -1:]

#     ctx = HeadAblator(ablation_spec) if ablation_spec else nullcontext()
#     with ctx:
#         first_tok, past_full = one_step_sdpa_no_attn(model, last_tok, past)
#         gen_txt = greedy_generate_from_past(model, tok, first_tok, past_full, max_new_tokens=max_new_tokens)

#     m = _LET_RE.search(gen_txt.strip())
#     pred = m.group(1) if m else None
#     return pred, gen_txt

# def eval_mcq(examples, ablation_spec=None, tag="baseline"):
#     rows = []
#     ok = 0
#     for i, ex in enumerate(examples):
#         pred, gen_txt = generate_mcq_letter(ex["prompt"], ablation_spec=ablation_spec, max_new_tokens=MAX_NEW_TOKENS)
#         got = int(pred == ex["gold"])
#         ok += got
#         rows.append({
#             "i": i,
#             "tag": tag,
#             "id": ex["id"],
#             "gold": ex["gold"],
#             "pred": pred,
#             "ok": got,
#             "prompt_len_tokens": ex["prompt_len_tokens"],
#             "truncated": ex["truncated"],
#             "domain": ex["domain"],
#             "sub_domain": ex["sub_domain"],
#             "difficulty": ex["difficulty"],
#             "length": ex["length"],
#             "gen_txt": gen_txt,
#         })
#     return {"acc": ok / max(1, len(examples)), "rows": rows}

# # Baseline
# res_base = eval_mcq(examples, ablation_spec=None, tag="baseline")
# print(f"[LongBench-v2 micro] baseline acc = {res_base['acc']:.3f}")

# # TopK ablation
# res_top = eval_mcq(examples, ablation_spec=spec_top, tag="topK")
# print(f"[LongBench-v2 micro] topK acc     = {res_top['acc']:.3f}  (drop={res_base['acc']-res_top['acc']:.3f})")

# # RandomK (repeat N_RAND draws)
# rand_accs = []
# rand_rows_all = []
# for r in range(N_RAND):
#     rp = sample_layer_matched_disjoint_random(_top_pairs, seed=4242 + r)
#     spec_r = spec_from_pairs(rp)
#     res_r = eval_mcq(examples, ablation_spec=spec_r, tag=f"randK_{r}")
#     rand_accs.append(res_r["acc"])
#     rand_rows_all.extend(res_r["rows"])
#     print(f"[LongBench-v2 micro] randK_{r} acc   = {res_r['acc']:.3f}")

# rand_mean = float(np.mean(rand_accs))
# rand_std  = float(np.std(rand_accs, ddof=0))
# print(f"[LongBench-v2 micro] randK mean±std = {rand_mean:.3f} ± {rand_std:.3f}")
# print(f"[LongBench-v2 micro] topK vs randK  = {res_top['acc']:.3f} vs {rand_mean:.3f}")

# # Save artifacts
# df_out = pd.DataFrame(res_base["rows"] + res_top["rows"] + rand_rows_all)
# out_csv = os.path.join(ART_DIR, "longbench_v2_microslice_results.csv")
# df_out.to_csv(out_csv, index=False)
# print("Saved:", out_csv)

# summary = pd.DataFrame([{
#     "dataset_source": lb_source,
#     "n_items": len(examples),
#     "max_encode_tokens": MAX_ENCODE,
#     "topK": len(_top_pairs),
#     "n_rand": N_RAND,
#     "baseline_acc": res_base["acc"],
#     "topK_acc": res_top["acc"],
#     "drop_topK": res_base["acc"] - res_top["acc"],
#     "randK_mean_acc": rand_mean,
#     "randK_std_acc": rand_std,
# }])
# summary_path = os.path.join(ART_DIR, "longbench_v2_microslice_summary.csv")
# summary.to_csv(summary_path, index=False)
# print("Saved:", summary_path)

# summary


# %%
# Cell 46 (REPLACE)
# ---------------------------
# LongBench v2 micro-slice external validation (STRICTLY NON-TRUNCATED)
# - SKIP examples that don't fit (no char-cut, no token trim)
# - Prefer 1 per sub_domain (then fill)
# - Baseline vs TopK vs layer-matched disjoint RandomK
# - Also teacher-forced gold logprob of the correct option (more stable for small N)
# ---------------------------

import os, re
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

# ---- requirements from earlier cells ----
assert "HeadAblator" in globals(), "Missing HeadAblator (run Day4 setup cells first)."
assert "prefill_build_kv" in globals(), "Missing prefill_build_kv (run KV prefill cell first)."
assert "one_step_sdpa_no_attn" in globals(), "Missing one_step_sdpa_no_attn (run restore helper cell)."
assert "greedy_generate_from_past" in globals(), "Missing greedy_generate_from_past (run restore helper cell)."
assert "tok" in globals() and "model" in globals(), "Missing tok/model."
model.eval()

# ---------------------------
# Config (tuned for A10G 24GB)
# ---------------------------
TARGET_N = 10
TOPK = 8
N_RAND = 10               # set 5 for quick, 10–20 overnight
MAX_NEW_TOKENS = 6
MAX_SCAN = 12000          # how many streamed items to inspect per cap
MAX_CONTEXT_CHARS_SKIP = 300000  # avoid pathological tokenization cost; skip if larger

MODEL_MAX = int(getattr(model.config, "max_position_embeddings", 32768))
base_cap = int(getattr(globals().get("cfg4", object()), "max_len_encode", 12000))
# On A10G, these are reasonable attempts without being reckless:
CAP_CANDIDATES = [base_cap, 16384, 20480, 24576]
CAP_CANDIDATES = [min(int(c), MODEL_MAX) for c in CAP_CANDIDATES]
CAP_CANDIDATES = sorted(list(dict.fromkeys([c for c in CAP_CANDIDATES if c > 0])))

CHUNK_SIZE = int(getattr(globals().get("cfg4", object()), "chunk_size", 512))
ART_DIR = globals().get("ART_DIR", os.path.expanduser("~/SageMaker/retrieval-head-atlas/artifacts"))
os.makedirs(ART_DIR, exist_ok=True)

# ---------------------------
# Load LongBench v2 (streaming)
# ---------------------------
try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("Please `pip install datasets` (or ensure it's installed) before running this cell.") from e

def _try_load_longbench_v2_stream():
    last_err = None
    for name in ["THUDM/LongBench-v2", "zai-org/LongBench-v2"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            return ds, name
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load LongBench v2 from HF. Last error: {last_err}")

# ---------------------------
# Prompt building
# ---------------------------
_LET_RE = re.compile(r"\b([ABCD])\b")

def build_mcq_prompt(context: str, question: str, A: str, B: str, C: str, D: str) -> str:
    # trailing space encourages single-letter continuation
    return (
        "You are given a long context and a multiple-choice question.\n"
        "Choose the correct option.\n"
        "Answer with ONLY a single letter: A, B, C, or D.\n\n"
        "=== CONTEXT START ===\n"
        f"{context}\n"
        "=== CONTEXT END ===\n\n"
        f"Question: {question}\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n\n"
        "Final answer: "
    )

def prompt_token_len(prompt: str) -> int:
    return len(tok(prompt, add_special_tokens=False).input_ids)

# ---------------------------
# Choose TopK heads (reuse top_pairs if present else from cand)
# ---------------------------
def spec_from_pairs(pairs):
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec

n_layers = int(globals().get("n_layers", getattr(model.config, "num_hidden_layers", 0) or 28))
n_heads  = int(globals().get("n_heads",  getattr(model.config, "num_attention_heads", 0) or 12))

if "top_pairs" in globals() and isinstance(globals()["top_pairs"], (list, tuple)) and len(globals()["top_pairs"]) > 0:
    top_pairs_use = [(int(L), int(H)) for (L, H) in globals()["top_pairs"][:TOPK]]
else:
    assert "cand" in globals(), "Need cand DataFrame from candidate-heads cell."
    top_pairs_use = [(int(cand.iloc[i]["layer"]), int(cand.iloc[i]["head"])) for i in range(min(TOPK, len(cand)))]

top_spec = spec_from_pairs(top_pairs_use)

def sample_layer_matched_disjoint_random(top_pairs, seed=0):
    rng = np.random.default_rng(seed)
    top_set = set((int(L), int(H)) for (L, H) in top_pairs)
    layer_counts = Counter([int(L) for (L, _) in top_pairs])

    out = []
    for L, k in layer_counts.items():
        banned = set(H for (LL, H) in top_set if LL == L)
        choices = [h for h in range(n_heads) if h not in banned]
        if len(choices) < k:
            # last resort, allow overlap if impossible (rare)
            choices = [h for h in range(n_heads)]
        picked = rng.choice(choices, size=k, replace=False).tolist()
        out.extend([(L, int(h)) for h in picked])

    # enforce global disjointness if possible
    out = [(L, H) for (L, H) in out if (L, H) not in top_set]
    while len(out) < len(top_pairs):
        L = int(rng.integers(0, n_layers))
        H = int(rng.integers(0, n_heads))
        if (L, H) in top_set or (L, H) in out:
            continue
        out.append((L, H))
    return out[:len(top_pairs)]

# ---------------------------
# Collect strictly non-truncated examples (prefer unique sub_domain first)
# ---------------------------
def collect_examples_strict_nontrunc(cap_tokens: int, target_n: int, max_scan: int):
    lb_stream, lb_source = _try_load_longbench_v2_stream()
    stats = Counter()
    chosen = []
    seen_sub = set()

    def maybe_add(ex, enforce_unique_sub: bool):
        context = ex.get("context", "") or ""
        question = ex.get("question", "") or ""
        A = ex.get("choice_A", "") or ""
        B = ex.get("choice_B", "") or ""
        C = ex.get("choice_C", "") or ""
        D = ex.get("choice_D", "") or ""
        gold = ex.get("answer", "") or ""
        subd = ex.get("sub_domain", None)

        if gold not in ["A", "B", "C", "D"]:
            stats["skip_bad_gold"] += 1
            return False
        if len(context) > MAX_CONTEXT_CHARS_SKIP:
            stats["skip_too_many_chars"] += 1
            return False
        if enforce_unique_sub and subd in seen_sub:
            stats["skip_dup_subdomain"] += 1
            return False

        prompt = build_mcq_prompt(context, question, A, B, C, D)
        L = prompt_token_len(prompt)
        if L > cap_tokens:
            stats["skip_too_long_tokens"] += 1
            return False

        chosen.append({
            "id": ex.get("_id", None),
            "domain": ex.get("domain", None),
            "sub_domain": subd,
            "difficulty": ex.get("difficulty", None),
            "length": ex.get("length", None),
            "prompt": prompt,
            "gold": gold,
            "prompt_len_tokens": int(L),
            "truncated": 0,
        })
        if subd is not None:
            seen_sub.add(subd)
        stats["kept"] += 1
        return True

    # pass 1: unique sub_domain
    scanned = 0
    for ex in lb_stream:
        scanned += 1
        if scanned > max_scan or len(chosen) >= target_n:
            break
        maybe_add(ex, enforce_unique_sub=True)

    # pass 2: fill remaining without unique constraint
    if len(chosen) < target_n:
        lb_stream2, _ = _try_load_longbench_v2_stream()
        scanned2 = 0
        for ex in lb_stream2:
            scanned2 += 1
            if scanned2 > max_scan or len(chosen) >= target_n:
                break
            maybe_add(ex, enforce_unique_sub=False)

    return chosen, lb_source, dict(stats)

examples = None
used_cap = None
used_source = None
used_stats = None

best = (0, None, None, None, None)
for cap in CAP_CANDIDATES:
    exs, src, stats = collect_examples_strict_nontrunc(int(cap), TARGET_N, MAX_SCAN)
    print(f"cap={cap}: collected {len(exs)}/{TARGET_N} strictly non-truncated (unique_sub={len(set(e['sub_domain'] for e in exs))})")
    if len(exs) > best[0]:
        best = (len(exs), exs, int(cap), src, stats)
    if len(exs) >= TARGET_N:
        break

_, examples, used_cap, used_source, used_stats = best
if examples is None or len(examples) < 3:
    raise RuntimeError(
        f"Could not collect >=3 strictly non-truncated examples under caps={CAP_CANDIDATES}. "
        f"Try increasing CAP_CANDIDATES (up to MODEL_MAX={MODEL_MAX}) or increase MAX_SCAN."
    )

print(f"Using cap={used_cap} from {used_source}. Final N={len(examples)}. Stats:", used_stats)
assert sum(e["truncated"] for e in examples) == 0

# ---------------------------
# Eval helpers: greedy letter + teacher-forced gold logprob(letter)
# ---------------------------
@torch.no_grad()
def one_step_sdpa_logits(model, token_id: torch.Tensor, past):
    restore = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    past_len = _cache_seq_len(past)
    attn_mask = torch.ones((token_id.shape[0], past_len + 1), device=model.device, dtype=torch.long)
    position_ids = torch.tensor([[past_len]], device=model.device, dtype=torch.long)

    out = model(
        input_ids=token_id,
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    logits = out.logits[:, -1, :].detach()
    _set_attn_impl(model, restore or "sdpa")
    return logits, out.past_key_values

@torch.no_grad()
def teacher_forced_logprob_letter(prompt: str, letter: str, ablation_spec=None):
    assert letter in ["A", "B", "C", "D"]
    ans_ids = tok(letter, add_special_tokens=False).input_ids
    if len(ans_ids) == 0:
        return float("nan")

    past, input_ids = prefill_build_kv(model, tok, prompt, max_len=used_cap, chunk_size=CHUNK_SIZE)

    # HARD assertion: no silent truncation
    expected = prompt_token_len(prompt)
    got = int(input_ids.shape[1])
    if got != expected:
        raise RuntimeError(
            f"Unexpected truncation in prefill_build_kv: expected_prompt_len={expected}, got_input_ids_len={got}. "
            f"Increase used_cap (currently {used_cap}) or ensure prompt fits."
        )

    last_tok = input_ids[:, -1:].long()

    ctx = HeadAblator(ablation_spec) if (ablation_spec is not None and len(ablation_spec) > 0) else nullcontext()
    with ctx:
        logits, past2 = one_step_sdpa_logits(model, last_tok, past)
        total = 0.0
        for tid in ans_ids:
            tid = int(tid)
            total += torch.log_softmax(logits, dim=-1)[0, tid].item()
            tok_in = torch.tensor([[tid]], device=model.device, dtype=torch.long)
            logits, past2 = one_step_sdpa_logits(model, tok_in, past2)
    return float(total)

@torch.no_grad()
def generate_mcq_letter(prompt: str, ablation_spec=None):
    past, input_ids = prefill_build_kv(model, tok, prompt, max_len=used_cap, chunk_size=CHUNK_SIZE)

    # HARD assertion: no silent truncation
    expected = prompt_token_len(prompt)
    got = int(input_ids.shape[1])
    if got != expected:
        raise RuntimeError(
            f"Unexpected truncation in prefill_build_kv: expected_prompt_len={expected}, got_input_ids_len={got}. "
            f"Increase used_cap (currently {used_cap}) or ensure prompt fits."
        )

    last_tok = input_ids[:, -1:].long()

    ctx = HeadAblator(ablation_spec) if (ablation_spec is not None and len(ablation_spec) > 0) else nullcontext()
    with ctx:
        first_tok, past_full = one_step_sdpa_no_attn(model, last_tok, past)
        gen_txt = greedy_generate_from_past(model, tok, first_tok, past_full, max_new_tokens=MAX_NEW_TOKENS)

    m = _LET_RE.search(gen_txt.strip().upper())
    pred = m.group(1) if m else ""
    return pred, gen_txt

def eval_examples(tag: str, ablation_spec=None):
    rows = []
    ok = 0
    gold_lps = []
    for i, ex in enumerate(examples):
        pred, gen_txt = generate_mcq_letter(ex["prompt"], ablation_spec=ablation_spec)
        got = int(pred == ex["gold"])
        ok += got

        gold_lp = teacher_forced_logprob_letter(ex["prompt"], ex["gold"], ablation_spec=ablation_spec)
        gold_lps.append(gold_lp)

        rows.append({
            "i": i,
            "tag": tag,
            "id": ex["id"],
            "gold": ex["gold"],
            "pred": pred,
            "ok": got,
            "gold_logprob": gold_lp,
            "prompt_len_tokens": ex["prompt_len_tokens"],
            "truncated": ex["truncated"],
            "domain": ex["domain"],
            "sub_domain": ex["sub_domain"],
            "difficulty": ex["difficulty"],
            "length": ex["length"],
            "gen_txt": gen_txt,
        })
    acc = ok / max(1, len(examples))
    mean_lp = float(np.mean(gold_lps)) if len(gold_lps) else float("nan")
    return acc, mean_lp, pd.DataFrame(rows)

# ---------------------------
# Run: baseline, TopK, RandomK draws
# ---------------------------
baseline_acc, baseline_lp, df_base = eval_examples("baseline", None)
top_acc, top_lp, df_top = eval_examples("topK", top_spec)

rand_accs = []
rand_lps = []
df_rands = []
for j in range(N_RAND):
    rpairs = sample_layer_matched_disjoint_random(top_pairs_use, seed=1000 + j)
    rspec = spec_from_pairs(rpairs)
    acc_j, lp_j, df_j = eval_examples(f"rand{j}", rspec)
    rand_accs.append(acc_j)
    rand_lps.append(lp_j)
    df_rands.append(df_j)

df_all = pd.concat([df_base, df_top] + df_rands, ignore_index=True)

summary = pd.DataFrame([{
    "dataset_source": used_source,
    "n_items": int(len(examples)),
    "unique_sub_domain": int(len(set(e["sub_domain"] for e in examples))),
    "max_encode_tokens_used": int(used_cap),
    "topK": int(TOPK),
    "n_rand": int(N_RAND),
    "baseline_acc": float(baseline_acc),
    "topK_acc": float(top_acc),
    "drop_topK_acc": float(baseline_acc - top_acc),
    "randK_mean_acc": float(np.mean(rand_accs)) if rand_accs else np.nan,
    "randK_std_acc": float(np.std(rand_accs)) if rand_accs else np.nan,
    "baseline_mean_gold_logprob": float(baseline_lp),
    "topK_mean_gold_logprob": float(top_lp),
    "delta_topK_mean_gold_logprob": float(top_lp - baseline_lp),
    "randK_mean_gold_logprob": float(np.mean(rand_lps)) if rand_lps else np.nan,
    "randK_std_gold_logprob": float(np.std(rand_lps)) if rand_lps else np.nan,
}])

print(f"[LongBench-v2 non-trunc] cap={used_cap} N={len(examples)} unique_sub={summary.loc[0,'unique_sub_domain']}")
print(f"Baseline acc={baseline_acc:.3f} | mean gold logprob={baseline_lp:.3f}")
print(f"TopK    acc={top_acc:.3f} (drop={baseline_acc-top_acc:.3f}) | Δmean gold logprob={top_lp-baseline_lp:.3f}")
print(f"RandK acc mean±std = {float(np.mean(rand_accs)):.3f} ± {float(np.std(rand_accs)):.3f}")
print(f"RandK gold logprob mean±std = {float(np.mean(rand_lps)):.3f} ± {float(np.std(rand_lps)):.3f}")

# Save
out_results = os.path.join(ART_DIR, "longbench_v2_nontrunc_microslice_results.csv")
out_summary  = os.path.join(ART_DIR, "longbench_v2_nontrunc_microslice_summary.csv")
df_all.to_csv(out_results, index=False)
summary.to_csv(out_summary, index=False)
print("Saved:", out_results)
print("Saved:", out_summary)

summary


# %%
# Cell 46
# ---------------------------
# Plot: distribution of Δlogprob improvements
# ---------------------------

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df_patch["delta_topK"], bins=20, alpha=0.8, label="patch topK")
ax.hist(df_patch["delta_randK"], bins=20, alpha=0.8, label="patch randomK")
ax.set_title(f"Day5: Patching effect on clean-answer logprob ({FAR_KEY})")
ax.set_xlabel("Δ mean logprob (patched - baseline)")
ax.set_ylabel("count")
ax.legend()

png = os.path.join(ART_DIR, "day5_patch_group_deltas.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", png)


# %%
# Cell 47
# ---------------------------
# Single-head patch scores (rank heads by sufficiency)
# ---------------------------

N_HEADS_TO_TEST = min(30, len(ranked_pairs))  # keep it sane
test_pairs = ranked_pairs[:N_HEADS_TO_TEST]

single_rows = []
for (L, H) in tqdm(test_pairs, desc="Day5 single-head patch"):
    spec_one = {int(L): [int(H)]}

    deltas = []
    for ex in paired:
        past_corrupt, last_tok_corrupt = prefill_last_tok(ex["prompt_corrupt"], cfg5.max_len_encode, cfg5.chunk_size)
        lp_base = mean_logprob_from_prefill(past_corrupt, last_tok_corrupt, ex["gold_clean"])

        clean_cache = capture_clean_cache(ex["prompt_clean"])
        lp_pat = mean_logprob_from_prefill(past_corrupt, last_tok_corrupt, ex["gold_clean"],
                                           patch_cache=clean_cache, patch_spec=spec_one)
        deltas.append(lp_pat - lp_base)

    deltas = np.array(deltas, dtype=np.float64)
    single_rows.append({
        "layer": int(L),
        "head": int(H),
        "mean_delta_logprob": float(deltas.mean()),
        "std_delta_logprob": float(deltas.std()),
    })

df_single = pd.DataFrame(single_rows).sort_values("mean_delta_logprob", ascending=False)
single_csv = os.path.join(ART_DIR, "day5_patch_single_head.csv")
df_single.to_csv(single_csv, index=False)
print("Saved:", single_csv)
df_single.head(15)


# %%
# Cell 44 (REPLACE)
# ---------------------------
# Choose which heads to patch (prefer ablation-ranked if available)
# AND build a DISTRIBUTION of layer-matched disjoint randomK patch specs
# ---------------------------
PATCH_SINGLE_PATH = os.path.join(ART_DIR, "day5_patch_single_head.csv")
if os.path.exists(PATCH_SINGLE_PATH):
    df_ps = pd.read_csv(PATCH_SINGLE_PATH).sort_values("mean_delta_logprob", ascending=False)
    ranked_pairs = list(zip(df_ps["layer"].astype(int), df_ps["head"].astype(int)))
    print("Using patch-sufficiency ranking:", PATCH_SINGLE_PATH)
else:
    # fall back to your current logic (logprob ablation ranking or cand)
    ...

import os
import numpy as np
import pandas as pd
from collections import Counter

CAND_PATH = os.path.join(ART_DIR, "day4_candidate_heads.csv")
assert os.path.exists(CAND_PATH), f"Missing {CAND_PATH} (run Day4 candidate selection first)."
cand = pd.read_csv(CAND_PATH)

FAR_KEY = f"L{cfg5.length}_p{cfg5.needle_frac:.2f}"
LOGP_PATH = os.path.join(ART_DIR, "day4_single_head_logprob.csv")

def spec_from_pairs(pairs):
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(map(int, spec[L])))
    return spec

# --- choose ranking source ---
if os.path.exists(LOGP_PATH):
    df_logp = pd.read_csv(LOGP_PATH)
    df_logp_far = df_logp[df_logp["setting"] == FAR_KEY].sort_values("delta_logprob")  # most negative = most important
    ranked_pairs = list(zip(df_logp_far["layer"].astype(int), df_logp_far["head"].astype(int)))
    print("Using logprob ablation ranking:", LOGP_PATH)
else:
    ranked_pairs = list(zip(cand["layer"].astype(int), cand["head"].astype(int)))
    print("Using Day4 candidate ranking:", CAND_PATH)

top_pairs = ranked_pairs[:cfg5.topK]
spec_top = spec_from_pairs(top_pairs)

# ---------------------------
# Random patch controls (distribution): layer-matched + disjoint from top_pairs
# ---------------------------

N_RAND_PATCH = 20  # reviewer request: 10–20
RANDOM_BASE_SEED = int(getattr(cfg5, "random_seed", 12345))

top_set = set((int(L), int(H)) for (L, H) in top_pairs)
layer_counts = Counter([int(L) for (L, _) in top_pairs])

def sample_layer_matched_disjoint_random_pairs(top_pairs, seed: int):
    rng = np.random.default_rng(int(seed))
    top_set_local = set((int(L), int(H)) for (L, H) in top_pairs)

    out = []
    # sample per-layer to match counts
    for L, k in layer_counts.items():
        banned = set(H for (LL, H) in top_set_local if LL == L)
        choices = [h for h in range(n_heads) if h not in banned]
        if len(choices) < k:
            # should be rare; fall back to any head in layer
            choices = list(range(n_heads))
        picked = rng.choice(choices, size=k, replace=False).tolist()
        out.extend([(int(L), int(h)) for h in picked])

    # enforce global disjointness + uniqueness
    out_unique = []
    seen = set()
    for p in out:
        if p in top_set_local:
            continue
        if p in seen:
            continue
        seen.add(p)
        out_unique.append(p)

    # refill if we lost any (keep disjoint from top_set)
    while len(out_unique) < len(top_pairs):
        L = int(rng.integers(0, n_layers))
        H = int(rng.integers(0, n_heads))
        p = (L, H)
        if p in top_set_local or p in seen:
            continue
        seen.add(p)
        out_unique.append(p)

    return out_unique[:len(top_pairs)]

rand_pairs_list = []
rand_specs_list = []
for r in range(N_RAND_PATCH):
    rp = sample_layer_matched_disjoint_random_pairs(top_pairs, seed=RANDOM_BASE_SEED + 1000 + r)
    rand_pairs_list.append(rp)
    rand_specs_list.append(spec_from_pairs(rp))

print("top_pairs:", top_pairs)
print("example rand_pairs[0]:", rand_pairs_list[0])
print(f"Built {len(rand_specs_list)} random patch specs (layer-matched, disjoint).")


# %%
# Debug Cell (add right after Cell 44)

from collections import Counter

def flatten_spec(spec):
    return sorted((int(L), int(H)) for L, hs in spec.items() for H in hs)

# 1) show layer-match counts
top_counts = Counter([L for (L, H) in top_pairs])
print("TopK layer counts:", dict(sorted(top_counts.items())))

for j in [0, 1, 2]:
    rp = rand_pairs_list[j]
    rc = Counter([L for (L, H) in rp])
    print(f"Rand{j} layer counts:", dict(sorted(rc.items())))

# 2) ensure disjoint from TopK
top_set = set((int(L), int(H)) for (L, H) in top_pairs)
viol = []
for i, rp in enumerate(rand_pairs_list):
    inter = top_set.intersection(set(rp))
    if len(inter) > 0:
        viol.append((i, list(inter)[:5]))
print("RandomK disjoint violations:", viol[:3], "count=", len(viol))

# 3) check how many unique random sets you actually got
as_fro = [frozenset(rp) for rp in rand_pairs_list]
print("Unique rand sets:", len(set(as_fro)), "/", len(as_fro))

# 4) print a few rand sets so you can eyeball variety
print("Example rand sets:")
for i in range(min(5, len(rand_pairs_list))):
    print(i, rand_pairs_list[i])


# %%
# Cell 45 (REPLACE)
# ---------------------------
# Run patching evaluation with RandomK as a distribution:
#  - patch topK once
#  - patch randK N_RAND_PATCH times (layer-matched, disjoint)
# Report mean±std across random draws (and also save per-draw deltas)
# ---------------------------

import os, json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

assert "rand_specs_list" in globals() and len(rand_specs_list) > 0, "Missing rand_specs_list (run Cell 44)."

rows = []

for ex in tqdm(paired, desc="Day5 patch eval (dist)"):
    # prefill corrupt ONCE
    past_corrupt, last_tok_corrupt = prefill_last_tok(ex["prompt_corrupt"], cfg5.max_len_encode, cfg5.chunk_size)

    # baseline: logprob of clean answer under corrupt prompt
    lp_base = mean_logprob_from_prefill(
        past_corrupt, last_tok_corrupt,
        gold=ex["gold_clean"],
        patch_cache=None,
        patch_spec=None
    )

    # capture clean cache (o_proj inputs at query step)
    clean_cache = capture_clean_cache(ex["prompt_clean"])

    # patch topK (single)
    lp_top = mean_logprob_from_prefill(
        past_corrupt, last_tok_corrupt,
        gold=ex["gold_clean"],
        patch_cache=clean_cache,
        patch_spec=spec_top
    )
    delta_top = float(lp_top - lp_base)

    # patch randomK (distribution)
    lp_rnds = []
    for spec_r in rand_specs_list:
        lp_r = mean_logprob_from_prefill(
            past_corrupt, last_tok_corrupt,
            gold=ex["gold_clean"],
            patch_cache=clean_cache,
            patch_spec=spec_r
        )
        lp_rnds.append(float(lp_r))

    lp_rnds = np.array(lp_rnds, dtype=np.float64)
    delta_rnds = lp_rnds - float(lp_base)

    rows.append({
        "i": ex["i"],
        "gold_clean": ex["gold_clean"],
        "gold_corrupt": ex["gold_corrupt"],
        "lp_base": float(lp_base),
        "lp_patch_topK": float(lp_top),
        "delta_topK": delta_top,

        # random distribution summaries
        "lp_patch_randK_mean": float(lp_rnds.mean()),
        "lp_patch_randK_std": float(lp_rnds.std(ddof=0)),
        "delta_randK_mean": float(delta_rnds.mean()),
        "delta_randK_std": float(delta_rnds.std(ddof=0)),

        # keep full per-draw deltas for later analysis/plots
        "delta_randK_all": json.dumps([float(x) for x in delta_rnds.tolist()]),
    })

df_patch = pd.DataFrame(rows)

out_csv = os.path.join(ART_DIR, "day5_patch_group_results.csv")
df_patch.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ---- report-grade summary: treat randomness as a distribution over draws ----
delta_mat = np.stack(df_patch["delta_randK_all"].apply(json.loads).to_list(), axis=0)  # [N_examples, N_RAND_PATCH]
rand_draw_means = delta_mat.mean(axis=0)  # mean across examples per draw

summary = {
    "N_examples": int(len(df_patch)),
    "N_RAND_PATCH": int(delta_mat.shape[1]),
    "topK_mean_delta": float(df_patch["delta_topK"].mean()),
    "topK_std_delta_across_examples": float(df_patch["delta_topK"].std(ddof=0)),
    "randK_draw_mean_of_means": float(rand_draw_means.mean()),
    "randK_draw_std_of_means": float(rand_draw_means.std(ddof=0)),
    "randK_all_deltas_mean": float(delta_mat.mean()),
    "randK_all_deltas_std": float(delta_mat.std(ddof=0)),
}
print("Day5 patch summary:", summary)

# optional: save per-draw means for plotting in report
draw_csv = os.path.join(ART_DIR, "day5_patch_rand_draw_means.csv")
pd.DataFrame({"draw": np.arange(len(rand_draw_means)), "mean_delta": rand_draw_means}).to_csv(draw_csv, index=False)
print("Saved:", draw_csv)

df_patch[["delta_topK","delta_randK_mean","delta_randK_std"]].describe()


# %%
# Cell 46 (REPLACE)
# ---------------------------
# Plot: TopK Δlogprob vs RandomK distribution (per-draw means)
# ---------------------------

import os, json
import numpy as np
import matplotlib.pyplot as plt

delta_mat = np.stack(df_patch["delta_randK_all"].apply(json.loads).to_list(), axis=0)
rand_draw_means = delta_mat.mean(axis=0)

top_mean = float(df_patch["delta_topK"].mean())

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(rand_draw_means, bins=15, alpha=0.85, label="RandomK draw means (Δlogprob)")
ax.axvline(top_mean, linewidth=2, label="TopK mean Δlogprob")
ax.set_title(f"Day5 patching: RandomK distribution vs TopK ({FAR_KEY})")
ax.set_xlabel("Δ mean logprob (patched - baseline)")
ax.set_ylabel("count")
ax.legend()

png = os.path.join(ART_DIR, "day5_patch_group_random_dist.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", png)

print("TopK mean Δ:", top_mean)
print("RandomK mean±std over draws:", float(rand_draw_means.mean()), float(rand_draw_means.std(ddof=0)))


# %%
# Cell 46 (REPLACE)
# ---------------------------
# Plot: RandomK distribution vs TopK (use a dot/strip plot instead of histogram)
# ---------------------------

import os, json
import numpy as np
import matplotlib.pyplot as plt

# Reconstruct [N_examples, N_RAND_PATCH] deltas from saved JSON lists
delta_mat = np.stack(df_patch["delta_randK_all"].apply(json.loads).to_list(), axis=0)
rand_draw_means = delta_mat.mean(axis=0)  # one value per random draw

top_mean = float(df_patch["delta_topK"].mean())
rand_mean = float(rand_draw_means.mean())
rand_std  = float(rand_draw_means.std(ddof=0))

# jitter in y so points are visible (no meaning on y-axis)
rng = np.random.default_rng(0)
y = rng.uniform(-0.06, 0.06, size=len(rand_draw_means))

fig, ax = plt.subplots(figsize=(7, 3.2))

# random draw means as points
ax.scatter(rand_draw_means, y, alpha=0.85, label="RandomK draw means (Δlogprob)")

# random mean ± std as a band
ax.axvspan(rand_mean - rand_std, rand_mean + rand_std, alpha=0.20, label="RandomK mean ± 1 std")
ax.axvline(rand_mean, linestyle="--", linewidth=2, label="RandomK mean")

# topK mean line
ax.axvline(top_mean, linewidth=3, label="TopK mean Δlogprob")

ax.set_title(f"Day5 patching: RandomK distribution vs TopK ({FAR_KEY})")
ax.set_xlabel("Δ mean logprob (patched - baseline)")
ax.set_yticks([])
ax.set_ylim(-0.12, 0.12)
ax.legend(loc="best")

png = os.path.join(ART_DIR, "day5_patch_group_random_dist.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", png)

print("TopK mean Δ:", top_mean)
print("RandomK mean±std over draws:", rand_mean, rand_std)
print("All RandomK draw means:", np.round(rand_draw_means, 4))


# %%
# Optional plot: per-example Δ distributions (TopK vs RandomK mean-per-example)

import json
import numpy as np
import matplotlib.pyplot as plt

delta_mat = np.stack(df_patch["delta_randK_all"].apply(json.loads).to_list(), axis=0)  # [N_examples, N_draws]
rand_per_ex_mean = delta_mat.mean(axis=1)  # [N_examples]
top_per_ex = df_patch["delta_topK"].to_numpy(dtype=np.float64)

fig, ax = plt.subplots(figsize=(7,4))
ax.hist(top_per_ex, bins=15, alpha=0.8, label="TopK Δ (per-example)")
ax.hist(rand_per_ex_mean, bins=15, alpha=0.8, label="RandomK mean Δ (per-example)")
ax.set_title(f"Day5 patching: per-example Δlogprob (TopK vs RandomK mean) ({FAR_KEY})")
ax.set_xlabel("Δ mean logprob (patched - baseline)")
ax.set_ylabel("count")
ax.legend()

png = os.path.join(ART_DIR, "day5_patch_group_per_example_hist.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", png)

print("TopK mean±std across examples:", float(top_per_ex.mean()), float(top_per_ex.std(ddof=0)))
print("RandomK mean±std across examples:", float(rand_per_ex_mean.mean()), float(rand_per_ex_mean.std(ddof=0)))

