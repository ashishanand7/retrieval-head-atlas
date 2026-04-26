import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rha.config import RuntimeConfig


def precision_to_dtype(precision: str):
    value = precision.lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    return "auto"


def load_model_and_tokenizer(
    cfg: RuntimeConfig,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    if cfg.hf_home:
        os.environ["HF_HOME"] = os.path.expanduser(cfg.hf_home)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=precision_to_dtype(cfg.precision),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def get_attn_impl(model) -> Optional[str]:
    return getattr(getattr(model, "config", None), "_attn_implementation", None)


def set_attn_impl(model, mode: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(mode)
    elif hasattr(model, "config"):
        model.config._attn_implementation = mode


@contextmanager
def use_attn_impl(model, mode: str):
    restore = get_attn_impl(model)
    set_attn_impl(model, mode)
    try:
        yield
    finally:
        set_attn_impl(model, restore or "sdpa")


def _get_layers(model) -> List[torch.nn.Module]:
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


def _get_self_attn(layer) -> torch.nn.Module:
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise RuntimeError("Could not locate attention module on a transformer layer.")


def _find_o_proj(attn, hidden_size: int) -> Tuple[torch.nn.Module, str]:
    for name in ["o_proj", "out_proj", "wo"]:
        if hasattr(attn, name):
            module = getattr(attn, name)
            if isinstance(module, torch.nn.Module):
                return module, name

    candidates = []
    for name, module in attn.named_modules():
        if isinstance(module, torch.nn.Linear) and getattr(module, "out_features", None) == hidden_size:
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("Could not find an attention output projection module.")

    candidates.sort(key=lambda item: (("o_proj" not in item[0] and "out" not in item[0] and "wo" not in item[0]), len(item[0])))
    name, module = candidates[0]
    return module, name


@dataclass
class ModelLayout:
    layers: List[torch.nn.Module]
    o_projs: List[torch.nn.Module]
    hidden_size: int
    num_heads: int
    head_dim: int


def inspect_model_layout(model) -> ModelLayout:
    layers = _get_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    head_dim = hidden_size // num_heads

    o_projs = []
    for layer in layers:
        attn = _get_self_attn(layer)
        module, _name = _find_o_proj(attn, hidden_size)
        o_projs.append(module)

    return ModelLayout(
        layers=layers,
        o_projs=o_projs,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
