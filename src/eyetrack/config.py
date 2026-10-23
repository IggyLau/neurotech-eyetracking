"""Load YAML config, layering an optional user file over config/default.yaml."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg
