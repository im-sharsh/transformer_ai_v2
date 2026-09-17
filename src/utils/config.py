"""Configuration loading with safe defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return deep_merge(yaml.safe_load(path.read_text()) or {}, overrides or {})
