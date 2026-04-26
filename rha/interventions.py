import random
from collections import Counter
from contextlib import AbstractContextManager
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from rha.modeling import ModelLayout

HeadPair = Tuple[int, int]
HeadSpec = Dict[int, List[int]]


def spec_from_pairs(pairs: Iterable[HeadPair]) -> HeadSpec:
    spec: HeadSpec = {}
    for layer, head in pairs:
        spec.setdefault(int(layer), []).append(int(head))
    return {layer: sorted(set(heads)) for layer, heads in spec.items()}


def pairs_from_spec(spec: HeadSpec) -> List[HeadPair]:
    return [(int(layer), int(head)) for layer, heads in spec.items() for head in heads]


def normalize_spec(spec: HeadSpec) -> HeadSpec:
    return {int(layer): sorted(set(map(int, heads))) for layer, heads in spec.items() if heads}


def layer_counts(pairs: Sequence[HeadPair]) -> Counter:
    return Counter(int(layer) for layer, _head in pairs)


def sample_layer_matched_disjoint(
    top_pairs: Sequence[HeadPair],
    num_heads_per_layer: int,
    seed: int,
    banned: Iterable[HeadPair] = (),
) -> List[HeadPair]:
    rng = random.Random(seed)
    banned_set = set((int(layer), int(head)) for layer, head in banned)
    counts = layer_counts(top_pairs)
    sampled: List[HeadPair] = []

    for layer, count in sorted(counts.items()):
        candidates = [
            (layer, head)
            for head in range(num_heads_per_layer)
            if (layer, head) not in banned_set
        ]
        if len(candidates) < count:
            raise ValueError(f"Layer {layer} has only {len(candidates)} available heads for {count} samples.")
        sampled.extend(rng.sample(candidates, count))

    return sampled


class HeadAblator(AbstractContextManager):
    """Zero selected head slices at each attention output projection input."""

    def __init__(self, layout: ModelLayout, spec: HeadSpec):
        self.layout = layout
        self.spec = normalize_spec(spec)
        self.handles = []

    def _make_pre_hook(self, layer_idx: int, heads_to_zero: List[int]):
        heads = torch.tensor(heads_to_zero, dtype=torch.long)

        def hook(_module, inputs):
            x_in = inputs[0]
            squeeze_back = False
            x = x_in
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_back = True

            if x.shape[-1] != self.layout.hidden_size:
                return None

            x2 = x.clone()
            batch, q_len, _hidden = x2.shape
            view = x2.reshape(batch, q_len, self.layout.num_heads, self.layout.head_dim)
            view.index_fill_(dim=2, index=heads.to(x2.device), value=0.0)
            x2 = view.reshape(batch, q_len, self.layout.hidden_size)

            if squeeze_back:
                x2 = x2.squeeze(1)
            return (x2,) + tuple(inputs[1:])

        return hook

    def __enter__(self):
        for layer_idx, heads in self.spec.items():
            if layer_idx < 0 or layer_idx >= len(self.layout.o_projs):
                raise ValueError(f"Layer index {layer_idx} is out of range.")
            for head in heads:
                if head < 0 or head >= self.layout.num_heads:
                    raise ValueError(f"Head index {head} is out of range for layer {layer_idx}.")
            handle = self.layout.o_projs[layer_idx].register_forward_pre_hook(
                self._make_pre_hook(layer_idx, heads)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False


class OProjInputCacher(AbstractContextManager):
    """Capture o_proj inputs by layer during a forward pass."""

    def __init__(self, layout: ModelLayout):
        self.layout = layout
        self.cache = {}
        self.handles = []

    def _hook(self, layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            self.cache[layer_idx] = x.detach().clone()
            return None

        return hook

    def __enter__(self):
        for layer_idx, module in enumerate(self.layout.o_projs):
            self.handles.append(module.register_forward_pre_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False


class HeadPatcher(AbstractContextManager):
    """Replace selected o_proj-input head slices with values from a clean cache."""

    def __init__(self, layout: ModelLayout, clean_cache: dict, spec: HeadSpec):
        self.layout = layout
        self.clean_cache = clean_cache
        self.spec = normalize_spec(spec)
        self.handles = []

    def _make_pre_hook(self, layer_idx: int, heads_to_patch: List[int]):
        heads = torch.tensor(heads_to_patch, dtype=torch.long)

        def hook(_module, inputs):
            if layer_idx not in self.clean_cache:
                return None

            x_in = inputs[0]
            squeeze_back = False
            x = x_in
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_back = True
            if x.shape[-1] != self.layout.hidden_size:
                return None

            x_clean = self.clean_cache[layer_idx].to(x.device)
            if x_clean.dim() == 2:
                x_clean = x_clean.unsqueeze(1)
            if x_clean.shape[-1] != self.layout.hidden_size:
                return None
            if x_clean.shape[0] != x.shape[0]:
                return None
            if x_clean.shape[1] != x.shape[1]:
                if x_clean.shape[1] < x.shape[1]:
                    return None
                x_clean = x_clean[:, -x.shape[1]:, :]

            x2 = x.clone()
            batch, q_len, _hidden = x2.shape
            view = x2.reshape(batch, q_len, self.layout.num_heads, self.layout.head_dim)
            clean_view = x_clean.reshape(batch, q_len, self.layout.num_heads, self.layout.head_dim)
            idx = heads.to(x2.device)
            view.index_copy_(dim=2, index=idx, source=clean_view.index_select(2, idx))
            x2 = view.reshape(batch, q_len, self.layout.hidden_size)

            if squeeze_back:
                x2 = x2.squeeze(1)
            return (x2,) + tuple(inputs[1:])

        return hook

    def __enter__(self):
        for layer_idx, heads in self.spec.items():
            if layer_idx < 0 or layer_idx >= len(self.layout.o_projs):
                raise ValueError(f"Layer index {layer_idx} is out of range.")
            handle = self.layout.o_projs[layer_idx].register_forward_pre_hook(
                self._make_pre_hook(layer_idx, heads)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False
