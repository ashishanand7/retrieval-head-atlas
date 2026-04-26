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
# Cell 3 
# Many models honor a config flag or method for attention implementation.
if hasattr(model, "set_attn_implementation"):
    model.set_attn_implementation("eager")
else:
    setattr(model.config, "attn_implementation", "eager")

prompt = "Say only: ready"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
        output_attentions=True,           # request attentions during generation
        return_dict_in_generate=True
    )

print("Generated text:", tok.decode(out.sequences[0], skip_special_tokens=True))
print("Attentions returned?", out.attentions is not None)
if out.attentions:
    # out.attentions = tuple[steps] of tuple[layers] of tensors [bs, heads, tgt, src]
    last_step = out.attentions[-1]
    print("Layers with attn:", len(last_step))
    print("One tensor shape:", last_step[0].shape)


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


# %%
# Cell 6 
def make_filler(n_words=1800, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{rng.integers(1_000_000)}" for _ in range(n_words)]
    return " ".join(words)

def make_needle_doc(L_tokens=2048, seed=0):
    # Reserve ~40 tokens for needle+Q/A; rest filler
    filler = make_filler(n_words=max(50, L_tokens-60), seed=seed)
    secret = 731992
    needle = f"[NEEDLE_START] The secret number is {secret}. [NEEDLE_END]"
    # q = "\nQ: What is the secret number?\nA:"
    q = "\nQ: What is the secret number?\nA (digits only): "
    return f"{filler}\n{needle}\n{q}", str(secret)

prompt, answer = make_needle_doc(L_tokens=2048, seed=42)
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False, return_dict_in_generate=True)
txt = tok.decode(out.sequences[0], skip_special_tokens=True)
print("-------------------------")
print("Last line", txt.splitlines()[-1])  # show the answer line

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
ART_DIR = os.path.expanduser("~/SageMaker/retrieval-head-atlas/artifacts")
os.makedirs(ART_DIR, exist_ok=True)

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# %%
# Cell 8 
NEEDLE_START = "[NEEDLE_START]"
NEEDLE_END   = "[NEEDLE_END]"

def _build_prompt_with_filler(filler_words:int, secret:int, seed:int):
    rng = np.random.default_rng(seed)
    words = [f"w{rng.integers(1_000_000)}" for _ in range(filler_words)]
    filler = " ".join(words)
    needle = f"{NEEDLE_START} The secret number is {secret}. {NEEDLE_END}"
    # q = "\nQ: What is the secret number?\nA:"
    q = "\nQ: What is the secret number?\nA (digits only): "
    return f"{filler}\n{needle}\n{q}"

# def make_needle_doc_fit(target_tokens:int=2048, secret:int=731992, seed:int=0, max_tok:int=8192):
#     """
#     Build a prompt whose *tokenized* length (with add_special_tokens=True) fits within max_tok,
#     leaving some margin so [NEEDLE] and the trailing 'A:' are not truncated.
#     """
#     # Start with a wide bracket for words; we'll binary-search the largest that fits.
#     lo, hi = 50, target_tokens * 3   # generous upper bound; tokenizer may split digits heavily
#     best = lo
#     for _ in range(16):  # ~binary search
#         mid = (lo + hi) // 2
#         p = _build_prompt_with_filler(mid, secret, seed)
#         ids = tok(p, add_special_tokens=True).input_ids
#         if len(ids) <= max_tok - 32:   # keep a small margin
#             best = mid
#             lo = mid + 1
#         else:
#             hi = mid - 1
#     prompt = _build_prompt_with_filler(best, secret, seed)
#     return prompt, str(secret)

def make_needle_doc_fit(target_tokens:int=2048, secret:int=731992, seed:int=0, max_tok:int=8192, margin:int=32):
    budget = min(target_tokens, max_tok) - margin
    lo, hi = 50, target_tokens * 4
    best = lo
    for _ in range(18):
        mid = (lo + hi) // 2
        p = _build_prompt_with_filler(mid, secret, seed)
        ids = tok(p, add_special_tokens=True).input_ids
        if len(ids) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    prompt = _build_prompt_with_filler(best, secret, seed)
    return prompt, str(secret)



# %%
# Cell 9
def _find_marker_variant_span(ids_list, marker_text: str):
    """
    Try several whitespace-aware encodings for a marker and return (start, end, which_variant) in token space.
    """
    variants = [
        marker_text,                 # bare
        " " + marker_text,           # space + marker
        "\n" + marker_text,          # newline + marker
        "\r\n" + marker_text,        # CRLF + marker (rare, but safe)
        "\t" + marker_text           # tab + marker
    ]
    for v in variants:
        v_ids = tok(v, add_special_tokens=False).input_ids
        s0, s1 = find_subsequence(ids_list, v_ids)
        if s0 != -1:
            return s0, s1, v
    return -1, -1, None



# %%
# Cell 10
from typing import List, Tuple

def find_subsequence(seq: List[int], subseq: List[int]) -> Tuple[int, int]:
    """Return (start, end) indices where seq[start:end] == subseq, or (-1, -1) if not found."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i, i + m
    return -1, -1

def locate_needle_span_in_ids(ids_list):
    """
    Robustly locate the token span strictly INSIDE [NEEDLE_START] ... [NEEDLE_END].
    Tries whitespace-aware variants; if still not found, falls back to phrase '.' boundary.
    """
    # 2.1 Try markers with whitespace-aware variants
    s0, s1, s_variant = _find_marker_variant_span(ids_list, NEEDLE_START)
    e0, e1, e_variant = _find_marker_variant_span(ids_list, NEEDLE_END)

    if s0 != -1 and e0 != -1 and e0 > s1:
        return s1, e0

    # 2.2 Fallback: use the literal phrase and next '.' if markers not found
    phrase = "The secret number is "
    # Also try whitespace-aware versions of the phrase (space/newline before)
    for pfx in ["", " ", "\n", "\r\n", "\t"]:
        phr_ids = tok(pfx + phrase, add_special_tokens=False).input_ids
        ps0, ps1 = find_subsequence(ids_list, phr_ids)
        if ps0 != -1:
            # Find a '.' token AFTER the phrase to mark the end of needle content
            dot_ids = tok(".", add_special_tokens=False).input_ids
            # Search dot beginning at ps1 (phrase end)
            pe0, pe1 = find_subsequence(ids_list[ps1:], dot_ids)
            if pe0 != -1:
                e_index = ps1 + pe0  # dot start index in ids_list
                # span is the answer digits between phrase and '.'
                return ps1, e_index

    # If we reach here, we genuinely failed to map the span
    raise RuntimeError("Failed to locate needle span (markers & phrase fallback).")


# %% [markdown]
# ### Prefill+one‑step attention readout

# %%
# Cell 11
def _set_attn_impl(model, mode: str):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(mode)
    else:
        model.config._attn_implementation = mode

def _cache_seq_len(past):
    if past is None:
        return 0
    # Newer HF cache objects
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    # Tuple-of-tuples: past[layer] = (k, v)
    k0 = past[0][0]
    # Usually [bs, heads, seq, head_dim]
    return int(k0.shape[-2])

@torch.no_grad()
def prefill_build_kv(model, tok, prompt: str, max_len: int, chunk_size: int = 1024):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", None)
    T = input_ids.shape[1]

    restore = getattr(model.config, "_attn_implementation", None)
    _set_attn_impl(model, "sdpa")

    past = None
    pos = 0
    while pos < T - 1:  # leave last token for the eager step
        end = min(T - 1, pos + chunk_size)
        chunk_ids = input_ids[:, pos:end]

        kwargs = dict(
            input_ids=chunk_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
        if attn_mask is not None:
            # IMPORTANT: prefix mask up to end (past + current)
            kwargs["attention_mask"] = attn_mask[:, :end]

        out = model(**kwargs)
        past = out.past_key_values
        pos = end

    # sanity check: cache should cover the whole prefix (T-1)
    got = _cache_seq_len(past)
    if got != T - 1:
        print(f"[warn] cache len {got} != expected {T-1} (are you on a windowed model?)")

    _set_attn_impl(model, restore or "sdpa")
    return past, input_ids



# %%
# Cell 12
@torch.no_grad()
def one_step_eager(model, last_token_id, past):
    restore = getattr(model.config, "_attn_implementation", None)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        setattr(model.config, "_attn_implementation", "eager")

    out = model(
        input_ids=last_token_id,
        use_cache=True, past_key_values=past,
        output_attentions=True, return_dict=True
    )
    logits = out.logits[:, -1, :]  # [1, vocab]
    next_token = logits.argmax(dim=-1, keepdim=True)

    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(restore or "sdpa")
    else:
        setattr(model.config, "_attn_implementation", restore or "sdpa")

    return next_token, logits, out.attentions, out.past_key_values


# %%
# Cell 13

# @dataclass
# class DetectConfig:
#     lengths: List[int] = (2048, 4096)     # keep modest today
#     n_prompts_per_len: int = 60           # bump later if you want tighter estimates
#     chunk_size: int = 1024
#     tau: float = 0.15                     # attention mass threshold on needle span
#     max_len_encode: int = 8192            # hard cap for tokenizer truncation
#     seed_base: int = 1000

from dataclasses import dataclass

@dataclass
class DetectConfig:
    lengths: list = (2048, 4096)     # keep modest for speed
    n_prompts_per_len: int = 60
    chunk_size: int = 1024
    tau: float = 0.05                # ↓ from 0.15 (more realistic mass cutoff)
    max_len_encode: int = 8192
    k_copy: int = 5                  # top-k for copy check
    seed_base: int = 1000


cfg = DetectConfig()

def get_model_dims(model):
    n_layers = getattr(model.config, "num_hidden_layers", None)
    n_heads  = getattr(model.config, "num_attention_heads", None)
    if n_layers is None or n_heads is None:
        raise RuntimeError("Could not read model dims.")
    return n_layers, n_heads

n_layers, n_heads = get_model_dims(model)

def score_heads_for_prompt(prompt: str, gold_answer: str) -> Dict:
    # Build KV with sdpa prefill, then one eager step to read attentions
    past, input_ids = prefill_build_kv(model, tok, prompt, max_len=cfg.max_len_encode, chunk_size=cfg.chunk_size)
    last_tok = input_ids[:, -1:]
    # next_tok, atts, _ = one_step_eager(model, last_tok, past)
    next_tok, logits, atts, _ = one_step_eager(model, last_tok, past)

    # Locate needle span in *input_ids* (key axis)
    ids_list = input_ids[0].tolist()
    s1, e0 = locate_needle_span_in_ids(ids_list)
    needle_idx = list(range(s1, e0))  # key positions

    gold_first_cands = set()
    for pfx in ["", " ", "\n"]:
        ids = tok(pfx + gold_answer, add_special_tokens=False).input_ids
        if ids:
            gold_first_cands.add(ids[0])
    
    copied_top1 = int(next_tok[0,0].item() in gold_first_cands)
    copied = int(next_tok[0,0].item() in gold_first_cands)
    
    topk_vals, topk_idx = torch.topk(logits, k=cfg.k_copy, dim=-1)
    copied_topk = any(int(i) in gold_first_cands for i in topk_idx[0].tolist())


    layer_masses = np.zeros((n_layers, n_heads), dtype=np.float32)
    if atts is not None and len(atts) == n_layers:
        T_abs = len(ids_list)  # full prompt length (absolute indices)
        for L in range(n_layers):
            a = atts[L][0]            # [H, 1, T_layer]
            a = a[:, 0, :]            # [H, T_layer]
            T_layer = a.shape[-1]
            # Start index of this layer's window inside the absolute token axis
            start = T_abs - T_layer
            # Map absolute needle indices -> local window indices; keep only those inside the window
            local = [j - start for j in needle_idx if 0 <= (j - start) < T_layer]
            if len(local) == 0:
                # This layer can't see the needle (window too short) -> mass is zero
                mass = torch.zeros(a.shape[0], device=a.device, dtype=a.dtype)
            else:
                idx = torch.as_tensor(local, device=a.device, dtype=torch.long)
                # Gather over the last dim and sum the mass across the needle span
                mass = a.index_select(dim=-1, index=idx).sum(dim=-1)  # [H]
            layer_masses[L, :] = mass.detach().cpu().float().numpy()

    else:
        # If for some reason we didn't get attentions, leave masses zero (will not pass tau)
        pass

    # binary hit per head
    hits = (layer_masses >= cfg.tau) & copied
    return {
        "masses": layer_masses,         # float array [L, H]
        "hits": hits.astype(np.int32),  # int array [L, H]
        "copied": copied
    }


# %%
# Cell 14

# DEBUG: single example check
secret = 654321
prompt, gold = make_needle_doc_fit(target_tokens=2048, secret=secret, seed=7, max_tok=cfg.max_len_encode)
past, input_ids = prefill_build_kv(model, tok, prompt, max_len=cfg.max_len_encode, chunk_size=cfg.chunk_size)
ids_list = input_ids[0].tolist()
s1, e0 = locate_needle_span_in_ids(ids_list)
print("needle span token indices:", s1, e0)
print("needle text (decoded):", tok.decode(ids_list[s1:e0]))
out = score_heads_for_prompt(prompt, gold)
print("copied?", out["copied"])


# %%
# Cell 15
def run_detection():
    rng = random.Random(cfg.seed_base)
    all_hits = np.zeros((n_layers, n_heads), dtype=np.int32)
    all_mass = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_total = 0

    for L in cfg.lengths:
        for i in trange(cfg.n_prompts_per_len, desc=f"L={L}"):
            secret = rng.randrange(100000, 999999)   # vary the needle value
            seed = cfg.seed_base + i + (L * 13)
            # prompt, gold = make_needle_doc(L_tokens=L, secret=secret, seed=seed)
            prompt, gold = make_needle_doc_fit(target_tokens=L, secret=secret, seed=seed, max_tok=cfg.max_len_encode)

            out = score_heads_for_prompt(prompt, gold)
            all_hits += out["hits"]
            all_mass += out["masses"]
            n_total += 1

    event_rate = all_hits / max(1, n_total)
    avg_mass   = all_mass / max(1, n_total)

    results = {
        "model_id": getattr(model, "name_or_path", "unknown"),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "lengths": list(cfg.lengths),
        "n_prompts_total": int(n_total),
        "tau": cfg.tau,
        "event_rate": event_rate.tolist(),   # 2D
        "avg_mass": avg_mass.tolist()        # 2D
    }
    path = os.path.join(ART_DIR, "head_scores.json")
    with open(path, "w") as f:
        json.dump(results, f)
    print("Saved:", path)
    return results

results = run_detection()


# %%
# Cell 16
import numpy as np, pandas as pd

er = np.array(results["event_rate"])  # [L, H]
am = np.array(results["avg_mass"])

# (Optional) layer-wise z-score of event_rate for emphasis
er_z = (er - er.mean(axis=1, keepdims=True)) / (er.std(axis=1, keepdims=True) + 1e-8)

rows = []
for L in range(results["n_layers"]):
    for H in range(results["n_heads"]):
        rows.append({
            "layer": L, "head": H,
            "event_rate": float(er[L, H]),
            "avg_mass": float(am[L, H]),
            "zscore_within_layer": float(er_z[L, H])
        })
df = pd.DataFrame(rows).sort_values(["zscore_within_layer","event_rate"], ascending=[False, False])
df.head(20)


# %%
# Cell 17
K = 30
topk = df.head(K)[["layer","head","event_rate","zscore_within_layer"]]
topk_path = os.path.join(ART_DIR, "topk_heads_working.csv")
topk.to_csv(topk_path, index=False)
topk_path


# %%
# Cell 18
import matplotlib.pyplot as plt
import numpy as np
import os

er = np.array(results["event_rate"])  # [n_layers, n_heads]

fig, ax = plt.subplots(figsize=(8, 6))

# Show heatmap: layers on y, heads on x
im = ax.imshow(er, aspect="auto", origin="upper")

ax.set_xlabel("head")
ax.set_ylabel("layer")

ax.set_xticks(np.arange(results["n_heads"]))
ax.set_xticklabels([f"h{h}" for h in range(results["n_heads"])])

ax.set_yticks(np.arange(results["n_layers"]))
ax.set_yticklabels(np.arange(results["n_layers"]))

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("event_rate")

fig.suptitle("Retrieval event rate per head (copy-paste metric)")

# Save a PNG instead of HTML
png_path = os.path.join(ART_DIR, "head_scores_heatmap.png")
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved to:", png_path)


# %%
# Cell 19
# ============================================================
# Day 3 — Long-range retrieval sweep (needle position) + metrics
# ============================================================

import os, re, json, math, random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import trange, tqdm
from typing import Dict, Tuple, List, Optional

model.eval()

# --------------- helpers: attention impl toggling ---------------

def _get_attn_impl(model):
    # best-effort getter across HF versions
    cfg = model.config
    if hasattr(cfg, "_attn_implementation") and cfg._attn_implementation is not None:
        return cfg._attn_implementation
    if hasattr(cfg, "attn_implementation") and cfg.attn_implementation is not None:
        return cfg.attn_implementation
    return None

def _set_attn_impl(model, mode: str):
    # best-effort setter across models/HF versions
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(mode)
        return
    # set both for safety
    try:
        model.config._attn_implementation = mode
    except Exception:
        pass
    try:
        model.config.attn_implementation = mode
    except Exception:
        pass

def _cache_seq_len(past) -> int:
    if past is None:
        return 0
    # HF cache object
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    # tuple-of-tuples style: past[layer] = (k, v)
    k0 = past[0][0]
    # usually [bs, heads, seq, dim] or [bs, seq, heads, dim]
    if k0.dim() == 4:
        # heuristic: seq is either -2 or -3 depending on layout
        # most common is [bs, heads, seq, dim] -> seq=-2
        return int(k0.shape[-2])
    raise RuntimeError("Unrecognized past_key_values format for seq_len.")

# --------------- subsequence + needle span location ---------------

def find_subsequence(seq: List[int], subseq: List[int]) -> Tuple[int, int]:
    n, m = len(seq), len(subseq)
    if m == 0 or m > n:
        return -1, -1
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i, i + m
    return -1, -1

def locate_answer_span_in_ids(ids_list: List[int], gold_answer: str, tok) -> Tuple[int, int]:
    # We prefer the span of the actual digits, not the whole marker block.
    # Try a few whitespace variants because tokenizers often bake whitespace into the first token.
    variants = [gold_answer, " " + gold_answer, "\n" + gold_answer, "\r\n" + gold_answer, "\t" + gold_answer]
    for v in variants:
        v_ids = tok(v, add_special_tokens=False).input_ids
        s0, s1 = find_subsequence(ids_list, v_ids)
        if s0 != -1:
            return s0, s1
    raise RuntimeError("Failed to locate gold answer span in prompt token IDs.")

# --------------- prompt builder with needle position sweep ---------------

NEEDLE_START = "[NEEDLE_START]"
NEEDLE_END   = "[NEEDLE_END]"

# digits-free filler vocabulary (prevents accidental number copying)
_WORDS = [
    "apple","river","candle","machine","window","planet","garden","music","paper","stone",
    "coffee","forest","animal","orange","silver","memory","camera","circle","future","ocean",
    "mountain","valley","library","puzzle","signal","engine","mirror","pocket","market","painter",
]

def _filler_words(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(_WORDS, size=n, replace=True).tolist())

def _build_prompt_positioned(prefix_words: int, suffix_words: int, secret: int, seed: int) -> str:
    # Instruction is important (and should remain at the top, not near the question).
    instr = (
        "You will be asked a question about the following document.\n"
        "Answer using ONLY the document. Respond with EXACTLY six digits and nothing else.\n"
    )
    prefix = _filler_words(prefix_words, seed)
    suffix = _filler_words(suffix_words, seed + 99991)

    needle = f"{NEEDLE_START} The secret number is {secret}. {NEEDLE_END}"

    q = "\nQuestion: What is the secret number?\nAnswer (six digits only): "
    return f"{instr}{prefix}\n{needle}\n{suffix}\n{q}"

def calibrate_total_filler_words_for_setting(
    tok,
    target_tokens: int,
    needle_frac: float,
    secret: int = 555555,
    seed: int = 0,
    margin_tokens: int = 96,
    max_tok_cap: int = 32768,
    bs_iters: int = 18,
) -> int:
    """
    Finds a total filler_words count such that tokenized prompt length <= (min(target_tokens, max_tok_cap) - margin).
    Done ONCE per (target_tokens, needle_frac) setting for speed.
    """
    budget = min(target_tokens, max_tok_cap) - margin_tokens
    if budget <= 64:
        raise ValueError("Budget too small; increase target_tokens or reduce margin_tokens.")

    lo, hi = 50, target_tokens * 12
    best = lo
    for _ in range(bs_iters):
        mid = (lo + hi) // 2
        pre = int(mid * needle_frac)
        suf = mid - pre
        p = _build_prompt_positioned(pre, suf, secret, seed)
        ids = tok(p, add_special_tokens=True).input_ids
        if len(ids) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def make_prompt_for_setting(
    tok,
    target_tokens: int,
    needle_frac: float,
    total_filler_words: int,
    secret: int,
    seed: int,
    margin_tokens: int = 96,
    max_tok_cap: int = 32768,
) -> Tuple[str, str]:
    """
    Build prompt using pre-calibrated total_filler_words. If a rare tokenization spike exceeds budget,
    it shrinks slightly and retries.
    """
    budget = min(target_tokens, max_tok_cap) - margin_tokens
    tw = total_filler_words

    for _ in range(4):
        pre = int(tw * needle_frac)
        suf = tw - pre
        prompt = _build_prompt_positioned(pre, suf, secret, seed)
        ids = tok(prompt, add_special_tokens=True).input_ids
        if len(ids) <= budget:
            return prompt, str(secret)
        tw = max(50, int(tw * 0.97))  # shrink a bit and retry

    # last resort: return whatever we have (still should fit in encode truncation if you set it)
    return prompt, str(secret)

# --------------- KV prefill (chunked, CORRECT cache accumulation) ---------------

@torch.no_grad()
def prefill_build_kv(
    model,
    tok,
    prompt: str,
    max_len: int,
    chunk_size: int = 1024,
):
    """
    Correct chunked prefill that accumulates past_key_values across chunks.
    Leaves the final prompt token to be processed in the eager step.
    """
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    input_ids = enc["input_ids"]
    T = input_ids.shape[1]
    if T < 2:
        raise ValueError("Prompt too short to prefill/measure.")

    attn_restore = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    past = None
    pos = 0
    while pos < T - 1:
        end = min(T - 1, pos + chunk_size)
        chunk_ids = input_ids[:, pos:end]
        past_len = _cache_seq_len(past)
        chunk_len = chunk_ids.shape[1]

        # Build attention mask of length (past + current). No padding here; all ones is fine.
        attn_mask = torch.ones((chunk_ids.shape[0], past_len + chunk_len), device=model.device, dtype=torch.long)

        # Provide position_ids to guarantee correct rotary positions in chunked mode.
        position_ids = torch.arange(past_len, past_len + chunk_len, device=model.device).unsqueeze(0)

        kwargs = dict(
            input_ids=chunk_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            attention_mask=attn_mask,
        )
        # Some implementations may not accept position_ids; try with, then without.
        try:
            out = model(**kwargs, position_ids=position_ids)
        except TypeError:
            out = model(**kwargs)

        past = out.past_key_values
        pos = end

    _set_attn_impl(model, attn_restore or "sdpa")
    return past, input_ids

@torch.no_grad()
def one_step_eager_with_attn(model, last_token_id: torch.Tensor, past):
    """
    Process the final prompt token in eager mode to obtain attention weights for that last position.
    Returns:
      next_token, logits, attentions, past_after_last_prompt_token
    """
    restore = _get_attn_impl(model)
    _set_attn_impl(model, "eager")

    past_len = _cache_seq_len(past)
    attn_mask = torch.ones((last_token_id.shape[0], past_len + 1), device=model.device, dtype=torch.long)
    position_ids = torch.tensor([[past_len]], device=model.device)

    kwargs = dict(
        input_ids=last_token_id,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
        output_attentions=True,
        attention_mask=attn_mask,
    )
    try:
        out = model(**kwargs, position_ids=position_ids)
    except TypeError:
        out = model(**kwargs)

    logits = out.logits[:, -1, :]
    next_token = logits.argmax(dim=-1, keepdim=True)

    _set_attn_impl(model, restore or "sdpa")
    return next_token, logits, out.attentions, out.past_key_values

# --------------- strict retrieval check (greedy decode, extract 6 digits) ---------------

_RE6 = re.compile(r"\d{6}")

def extract_6digits(text: str) -> Optional[str]:
    m = _RE6.search(text)
    return m.group(0) if m else None

@torch.no_grad()
def greedy_generate_from_past(model, tok, first_token: torch.Tensor, past, max_new_tokens: int = 12) -> str:
    """
    Greedily generate max_new_tokens starting with first_token, using sdpa (fast).
    Returns decoded generated text (only the generated tokens, not the prompt).
    """
    restore = _get_attn_impl(model)
    _set_attn_impl(model, "sdpa")

    toks = [first_token]
    cur = first_token
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=cur, past_key_values=past, use_cache=True, return_dict=True, output_attentions=False)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        toks.append(cur)

    _set_attn_impl(model, restore or "sdpa")

    gen_ids = torch.cat(toks, dim=1)[0].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True)

# --------------- scoring one prompt: masses + detection + events + retrieval ---------------

@dataclass
class Day3Config:
    lengths: Tuple[int, ...] = (2048, 4096, 8192)
    needle_fracs: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_prompts_per_setting: int = 60
    tau: float = 0.05
    chunk_size: int = 1024
    max_len_encode: int = 16384
    margin_tokens: int = 96
    max_new_tokens: int = 14
    seed_base: int = 1000

cfg3 = Day3Config()

def score_prompt_day3(prompt: str, gold_answer: str) -> Dict:
    past, input_ids = prefill_build_kv(
        model, tok, prompt,
        max_len=cfg3.max_len_encode,
        chunk_size=cfg3.chunk_size
    )

    # last prompt token is processed in eager mode to read attentions
    last_tok = input_ids[:, -1:]
    next_tok, logits, atts, past_after_last = one_step_eager_with_attn(model, last_tok, past)

    # strict retrieval check
    gen_txt = greedy_generate_from_past(model, tok, next_tok, past_after_last, max_new_tokens=cfg3.max_new_tokens)
    extracted = extract_6digits(gen_txt)
    retrieved = int(extracted == gold_answer)

    # locate gold answer span in the prompt token IDs (keys axis)
    ids_list = input_ids[0].tolist()
    ans_s0, ans_s1 = locate_answer_span_in_ids(ids_list, gold_answer, tok)
    needle_idx_abs = list(range(ans_s0, ans_s1))

    # distance from answer span to the query position (last prompt token index)
    T_abs = len(ids_list)
    distance_tokens = (T_abs - 1) - ans_s1  # approx tokens between end of digits and last prompt token

    # attention mass toward the needle span for the last prompt position
    layer_masses = np.zeros((n_layers, n_heads), dtype=np.float32)

    if atts is None or len(atts) != n_layers:
        # should not happen; leave zeros
        pass
    else:
        for L in range(n_layers):
            # atts[L] shape: [bs, heads, tgt=1, src]
            a = atts[L][0]      # [heads, 1, src]
            a = a[:, 0, :]      # [heads, src]
            T_layer = a.shape[-1]
            start_abs = T_abs - T_layer  # sliding window offset (0 if full)
            local = [j - start_abs for j in needle_idx_abs if 0 <= (j - start_abs) < T_layer]
            if len(local) == 0:
                continue
            idx = torch.as_tensor(local, device=a.device, dtype=torch.long)
            mass = a.index_select(dim=-1, index=idx).sum(dim=-1)  # [heads]
            layer_masses[L, :] = mass.detach().cpu().float().numpy()

    detect = (layer_masses >= cfg3.tau).astype(np.int32)          # per-head detection (attn mass large enough)
    events = (detect & retrieved).astype(np.int32)               # per-head event (detect AND retrieved)

    # optional: quick top-1 first-token match diagnostic (not used for metrics)
    gold_first_cands = set()
    for pfx in ["", " ", "\n", "\r\n", "\t"]:
        ids = tok(pfx + gold_answer, add_special_tokens=False).input_ids
        if ids:
            gold_first_cands.add(ids[0])
    copied_first = int(next_tok[0, 0].item() in gold_first_cands)

    return dict(
        masses=layer_masses,
        detect=detect,
        events=events,
        retrieved=retrieved,
        copied_first=copied_first,
        extracted=extracted,
        prompt_len=T_abs,
        distance_tokens=int(distance_tokens),
    )

# --------------- sweep runner + artifact generation ---------------

def _setting_key(L: int, frac: float) -> str:
    return f"L{L}_p{frac:.2f}"

def plot_heatmap(mat: np.ndarray, title: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect="auto", origin="upper")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"h{h}" for h in range(mat.shape[1])])
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(np.arange(mat.shape[0]))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")
    fig.suptitle(title)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def run_day3_sweep(cfg3: Day3Config) -> Dict:
    rng = random.Random(cfg3.seed_base)

    # pre-calibrate filler word counts per setting to avoid per-prompt binary search
    filler_words_by_setting = {}
    for L in cfg3.lengths:
        for frac in cfg3.needle_fracs:
            tw = calibrate_total_filler_words_for_setting(
                tok=tok,
                target_tokens=L,
                needle_frac=frac,
                secret=555555,
                seed=0,
                margin_tokens=cfg3.margin_tokens,
                max_tok_cap=cfg3.max_len_encode,
            )
            filler_words_by_setting[_setting_key(L, frac)] = int(tw)

    out = {
        "model_id": getattr(model, "name_or_path", "unknown"),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "tau": float(cfg3.tau),
        "lengths": list(map(int, cfg3.lengths)),
        "needle_fracs": list(map(float, cfg3.needle_fracs)),
        "n_prompts_per_setting": int(cfg3.n_prompts_per_setting),
        "chunk_size": int(cfg3.chunk_size),
        "max_len_encode": int(cfg3.max_len_encode),
        "margin_tokens": int(cfg3.margin_tokens),
        "max_new_tokens": int(cfg3.max_new_tokens),
        "settings": {}
    }

    setting_rows = []  # model-level per setting summary

    for L in cfg3.lengths:
        for frac in cfg3.needle_fracs:
            key = _setting_key(L, frac)
            total_words = filler_words_by_setting[key]

            detect_cnt = np.zeros((n_layers, n_heads), dtype=np.int64)
            event_cnt  = np.zeros((n_layers, n_heads), dtype=np.int64)
            mass_sum   = np.zeros((n_layers, n_heads), dtype=np.float64)

            retrieved_cnt = 0
            copied_first_cnt = 0
            prompt_len_sum = 0
            dist_sum = 0

            for i in trange(cfg3.n_prompts_per_setting, desc=f"{key}"):
                secret = rng.randrange(100000, 999999)
                seed = cfg3.seed_base + (L * 10000) + int(frac * 1000) + i

                prompt, gold = make_prompt_for_setting(
                    tok=tok,
                    target_tokens=L,
                    needle_frac=frac,
                    total_filler_words=total_words,
                    secret=secret,
                    seed=seed,
                    margin_tokens=cfg3.margin_tokens,
                    max_tok_cap=cfg3.max_len_encode,
                )

                res = score_prompt_day3(prompt, gold)

                detect_cnt += res["detect"]
                event_cnt  += res["events"]
                mass_sum   += res["masses"]

                retrieved_cnt += int(res["retrieved"])
                copied_first_cnt += int(res["copied_first"])
                prompt_len_sum += int(res["prompt_len"])
                dist_sum += int(res["distance_tokens"])

            n = cfg3.n_prompts_per_setting
            detect_rate = detect_cnt / max(1, n)
            event_rate  = event_cnt  / max(1, n)
            avg_mass    = mass_sum   / max(1, n)

            retrieval_rate = retrieved_cnt / max(1, n)
            # Retrieval score per head: P(detect | retrieved) = event_rate / retrieval_rate
            # (when retrieval_rate is 0, set score to 0 to avoid NaNs)
            if retrieval_rate > 0:
                retrieval_score = event_rate / retrieval_rate
            else:
                retrieval_score = np.zeros_like(event_rate, dtype=np.float64)

            avg_prompt_len = prompt_len_sum / max(1, n)
            avg_distance   = dist_sum / max(1, n)

            out["settings"][key] = {
                "target_tokens": int(L),
                "needle_frac": float(frac),
                "total_filler_words": int(total_words),
                "retrieval_rate": float(retrieval_rate),
                "copied_first_rate": float(copied_first_cnt / max(1, n)),
                "avg_prompt_len": float(avg_prompt_len),
                "avg_distance_tokens": float(avg_distance),
                "detect_rate": detect_rate.tolist(),
                "event_rate": event_rate.tolist(),
                "retrieval_score": retrieval_score.tolist(),
                "avg_mass": avg_mass.tolist(),
            }

            setting_rows.append({
                "setting": key,
                "length": int(L),
                "needle_frac": float(frac),
                "total_filler_words": int(total_words),
                "retrieval_rate": float(retrieval_rate),
                "copied_first_rate": float(copied_first_cnt / max(1, n)),
                "avg_prompt_len": float(avg_prompt_len),
                "avg_distance_tokens": float(avg_distance),
            })

            # Save heatmaps per setting (event_rate + retrieval_score are usually the most informative)
            plot_heatmap(event_rate, f"Event rate (detect & retrieved) — {key}", os.path.join(ART_DIR, f"day3_event_rate_{key}.png"))
            plot_heatmap(detect_rate, f"Detection rate (mass>=tau) — {key}", os.path.join(ART_DIR, f"day3_detect_rate_{key}.png"))
            plot_heatmap(retrieval_score, f"Retrieval score P(detect|retrieved) — {key}", os.path.join(ART_DIR, f"day3_retrieval_score_{key}.png"))

    # write main json
    json_path = os.path.join(ART_DIR, "day3_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f)
    print("Saved:", json_path)

    # model-level setting summary
    df_settings = pd.DataFrame(setting_rows).sort_values(["length", "needle_frac"])
    settings_csv = os.path.join(ART_DIR, "day3_setting_summary.csv")
    df_settings.to_csv(settings_csv, index=False)
    print("Saved:", settings_csv)

    return out, df_settings

day3_results, df_setting_summary = run_day3_sweep(cfg3)
df_setting_summary


# %%
# Cell 20
# ==========================================
# Day 3 — Flatten results to per-head dataframe
# ==========================================

def results_to_long_df(day3_results: Dict) -> pd.DataFrame:
    rows = []
    for key, item in day3_results["settings"].items():
        L = int(item["target_tokens"])
        frac = float(item["needle_frac"])
        retrieval_rate = float(item["retrieval_rate"])
        avg_prompt_len = float(item["avg_prompt_len"])
        avg_dist = float(item["avg_distance_tokens"])

        detect = np.array(item["detect_rate"])
        event  = np.array(item["event_rate"])
        score  = np.array(item["retrieval_score"])
        mass   = np.array(item["avg_mass"])

        for layer in range(detect.shape[0]):
            for head in range(detect.shape[1]):
                rows.append({
                    "setting": key,
                    "length": L,
                    "needle_frac": frac,
                    "retrieval_rate": retrieval_rate,
                    "avg_prompt_len": avg_prompt_len,
                    "avg_distance_tokens": avg_dist,
                    "layer": layer,
                    "head": head,
                    "detect_rate": float(detect[layer, head]),
                    "event_rate": float(event[layer, head]),
                    "retrieval_score": float(score[layer, head]),
                    "avg_mass": float(mass[layer, head]),
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
# ===================================================

def distance_sensitivity_report(df_long: pd.DataFrame, far_max: float = 0.3, near_min: float = 0.7) -> pd.DataFrame:
    far = df_long[df_long["needle_frac"] <= far_max].groupby(["layer","head"], as_index=False).agg(
        far_event=("event_rate","mean"),
        far_detect=("detect_rate","mean"),
        far_score=("retrieval_score","mean"),
        far_rr=("retrieval_rate","mean"),
    )
    near = df_long[df_long["needle_frac"] >= near_min].groupby(["layer","head"], as_index=False).agg(
        near_event=("event_rate","mean"),
        near_detect=("detect_rate","mean"),
        near_score=("retrieval_score","mean"),
        near_rr=("retrieval_rate","mean"),
    )
    m = far.merge(near, on=["layer","head"], how="inner")
    m["delta_event_far_minus_near"] = m["far_event"] - m["near_event"]
    m["delta_score_far_minus_near"] = m["far_score"] - m["near_score"]
    m["event_ratio_far_over_near"] = (m["far_event"] + 1e-9) / (m["near_event"] + 1e-9)
    m["score_ratio_far_over_near"] = (m["far_score"] + 1e-9) / (m["near_score"] + 1e-9)
    return m.sort_values(["delta_event_far_minus_near", "delta_score_far_minus_near"], ascending=False)

df_dist = distance_sensitivity_report(df_long, far_max=0.3, near_min=0.7)
dist_csv = os.path.join(ART_DIR, "day3_distance_sensitivity_heads.csv")
df_dist.to_csv(dist_csv, index=False)
print("Saved:", dist_csv)

df_dist.head(30)


# %%
# Cell 22
# ==========================================
# Day 3 — quick sanity prints
# ==========================================

print(df_setting_summary[["length","needle_frac","retrieval_rate","copied_first_rate","avg_prompt_len","avg_distance_tokens"]]
      .sort_values(["length","needle_frac"])
      .to_string(index=False))


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
# 6) Candidate head selection from Day 3 sweep JSON
# ---------------------------

DAY3_JSON = os.path.join(ART_DIR, "day3_sweep_results.json")
assert os.path.exists(DAY3_JSON), f"Missing {DAY3_JSON} (run Day 3 first)."
day3 = json.load(open(DAY3_JSON, "r"))

def day3_long_df_from_json(day3: Dict) -> pd.DataFrame:
    rows = []
    for setting, item in day3["settings"].items():
        L = int(item["target_tokens"])
        frac = float(item["needle_frac"])
        rr = float(item["retrieval_rate"])
        detect = np.array(item["detect_rate"], dtype=float)
        event  = np.array(item["event_rate"], dtype=float)
        score  = np.array(item["retrieval_score"], dtype=float)

        for layer in range(detect.shape[0]):
            for head in range(detect.shape[1]):
                rows.append({
                    "setting": setting,
                    "length": L,
                    "needle_frac": frac,
                    "retrieval_rate": rr,
                    "layer": layer,
                    "head": head,
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
    # Prefer heads that still work far away (not just "wins by being bad near")
    m["rank_key"] = m["far_event"] + 0.5*m["far_score"]
    return m.sort_values(["rank_key","far_event","far_score"], ascending=False)

df_sens = distance_sensitivity(df3, far_max=cfg4.far_max, near_min=cfg4.near_min)
cand = df_sens.head(cfg4.top_n_heads)[["layer","head","far_event","far_score","near_event","near_score","rank_key"]].copy()
cand_path = os.path.join(ART_DIR, f"{cfg4.out_prefix}_candidate_heads.csv")
cand.to_csv(cand_path, index=False)
print("Saved:", cand_path)
cand.head(15)


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
# 8) Group ablations (top-K) + random control
# ---------------------------

def spec_from_pairs(pairs: Iterable[Tuple[int,int]]) -> Dict[int, List[int]]:
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    # dedupe
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec

def sample_random_heads(n: int, seed: int = 0) -> List[Tuple[int,int]]:
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n:
        pairs.add((int(rng.integers(0, n_layers)), int(rng.integers(0, n_heads))))
    return list(pairs)

topK = min(10, len(cand))
top_pairs = [(int(cand.iloc[i]["layer"]), int(cand.iloc[i]["head"])) for i in range(topK)]
spec_top = spec_from_pairs(top_pairs)
spec_rand = spec_from_pairs(sample_random_heads(topK, seed=4242))

group_rows = []
for key, ds in datasets.items():
    base_rr = float(baseline[key]["retrieval_rate"])
    print("getting dataset")
    rr_top = eval_dataset(ds, ablation_spec=spec_top)["retrieval_rate"]
    rr_rnd = eval_dataset(ds, ablation_spec=spec_rand)["retrieval_rate"]

    group_rows.append({
        "setting": key,
        "baseline_rr": base_rr,
        "topK": topK,
        "ablated_topK_rr": rr_top,
        "drop_topK_rr": base_rr - rr_top,
        "ablated_randK_rr": rr_rnd,
        "drop_randK_rr": base_rr - rr_rnd,
        "top_pairs": top_pairs,
    })
    print(f"{key} GROUP ablate topK drop={base_rr-rr_top:.3f} | randomK drop={base_rr-rr_rnd:.3f}")

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
# Cell 34
import numpy as np

def prompt_len_and_distance(prompt: str, gold: str):
    ids = tok(prompt, add_special_tokens=True).input_ids
    s0, s1 = locate_answer_span_in_ids(ids, gold, tok)  # from Day 3
    dist = (len(ids) - 1) - s1
    return len(ids), dist

for k, ds in datasets.items():
    lens, dists = [], []
    for ex in ds:
        L, D = prompt_len_and_distance(ex["prompt"], ex["gold"])
        lens.append(L); dists.append(D)
    lens = np.array(lens); dists = np.array(dists)
    print(f"{k}: prompt_len mean={lens.mean():.1f} (min={lens.min()}, max={lens.max()}), "
          f"distance mean={dists.mean():.1f} (min={dists.min()}, max={dists.max()})")


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
# Group log-prob ablations (topK vs randomK)
# ---------------------------

topK = min(10, len(cand))
top_pairs = [(int(cand.iloc[i]["layer"]), int(cand.iloc[i]["head"])) for i in range(topK)]

def spec_from_pairs(pairs):
    spec = {}
    for L, H in pairs:
        spec.setdefault(int(L), []).append(int(H))
    for L in list(spec.keys()):
        spec[L] = sorted(set(spec[L]))
    return spec

def sample_random_pairs(n: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n:
        pairs.add((int(rng.integers(0, n_layers)), int(rng.integers(0, n_heads))))
    return list(pairs)

spec_top  = spec_from_pairs(top_pairs)
spec_rand = spec_from_pairs(sample_random_pairs(topK, seed=4242))

group_logp_rows = []
for key, ds in datasets.items():
    base = eval_logprob_dataset(ds, None)
    top  = eval_logprob_dataset(ds, spec_top)
    rnd  = eval_logprob_dataset(ds, spec_rand)

    group_logp_rows.append({
        "setting": key,
        "topK": topK,
        "baseline_mean_logprob": base["mean_logprob"],
        "topK_mean_logprob": top["mean_logprob"],
        "randK_mean_logprob": rnd["mean_logprob"],
        "delta_topK": top["mean_logprob"] - base["mean_logprob"],
        "delta_randK": rnd["mean_logprob"] - base["mean_logprob"],
    })
    print(f"{key}: Δlogp_topK={top['mean_logprob']-base['mean_logprob']:.3f}, "
          f"Δlogp_randK={rnd['mean_logprob']-base['mean_logprob']:.3f}")

df_logp_group = pd.DataFrame(group_logp_rows)
logp_group_csv = os.path.join(ART_DIR, f"{cfg4.out_prefix}_group_logprob.csv")
df_logp_group.to_csv(logp_group_csv, index=False)
print("Saved:", logp_group_csv)
df_logp_group


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
# Choose which heads to patch (prefer ablation-ranked if available)
# ---------------------------

CAND_PATH = os.path.join(ART_DIR, "day4_candidate_heads.csv")
assert os.path.exists(CAND_PATH), f"Missing {CAND_PATH} (run Day4 candidate selection first)."
cand = pd.read_csv(CAND_PATH)

FAR_KEY = f"L{cfg5.length}_p{cfg5.needle_frac:.2f}"
LOGP_PATH = os.path.join(ART_DIR, "day4_single_head_logprob.csv")

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

# random control with same number of heads
rng = np.random.default_rng(cfg5.random_seed)
rand_pairs = []
seen = set()
while len(rand_pairs) < cfg5.topK:
    p = (int(rng.integers(0, n_layers)), int(rng.integers(0, n_heads)))
    if p not in seen:
        seen.add(p)
        rand_pairs.append(p)
spec_rand = spec_from_pairs(rand_pairs)

print("top_pairs:", top_pairs)
print("rand_pairs:", rand_pairs)


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

