from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    precision: str = "auto"
    context_hint_tokens: int = 8192
    hf_home: Optional[str] = None


def load_runtime_config(path: Optional[str] = None) -> RuntimeConfig:
    cfg_path = Path(path) if path else repo_root() / "config.yaml"
    if not cfg_path.exists():
        return RuntimeConfig()

    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return RuntimeConfig(
        model_id=str(raw.get("model_id", RuntimeConfig.model_id)),
        precision=str(raw.get("precision", RuntimeConfig.precision)),
        context_hint_tokens=int(raw.get("context_hint_tokens", RuntimeConfig.context_hint_tokens)),
        hf_home=raw.get("hf_home"),
    )


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate
