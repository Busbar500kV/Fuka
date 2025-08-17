# core/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import json
import os

# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 1000
    noise_sigma: float = 0.01
    # sources: list of dicts, each like:
    # {"kind":"moving_peak","amp":1.0,"speed":0.10,"width":4.0,"start":24}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 4.0, "start": 100}
    ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64
    k_flux: float = 0.05
    k_motor: float = 2.0
    diffuse: float = 0.05
    decay: float = 0.01
    env: FieldCfg = field(default_factory=FieldCfg)

# -----------------------------
# Defaults loader / deep-merge
# -----------------------------

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (src wins). Returns dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_defaults_json(path: str = "defaults.json") -> Dict[str, Any]:
    """Load defaults.json from repo root; return {} if missing/invalid."""
    try:
        # If caller passed an absolute/relative path, use it as-is
        cand = path
        if not os.path.isabs(cand):
            # Resolve relative to CWD (Streamlit runs from project root)
            cand = os.path.join(os.getcwd(), path)
        with open(cand, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# -----------------------------
# Public API (unchanged names)
# -----------------------------

def default_config() -> Dict[str, Any]:
    """
    Return a plain dict of defaults, with values from defaults.json
    (if present) merged on top. This keeps app.py unchanged.
    """
    base = asdict(Config())
    file_overrides = _load_defaults_json("defaults.json")
    return _deep_merge(base, file_overrides)

def make_config_from_dict(d: Dict[str, Any]) -> Config:
    """
    Build a Config dataclass from a (possibly merged) dict. This is used
    right before creating the Engine, so app/state edits just work.
    """
    env_d = d.get("env", {})
    fcfg = FieldCfg(
        length=int(env_d.get("length", 512)),
        frames=int(env_d.get("frames", d.get("frames", 5000))),
        noise_sigma=float(env_d.get("noise_sigma", 0.01)),
        sources=env_d.get("sources", FieldCfg().sources),
    )
    return Config(
        seed=int(d.get("seed", 0)),
        frames=int(d.get("frames", 5000)),
        space=int(d.get("space", 64)),
        k_flux=float(d.get("k_flux", 0.05)),
        k_motor=float(d.get("k_motor", 2.0)),
        diffuse=float(d.get("diffuse", 0.05)),
        decay=float(d.get("decay", 0.01)),
        env=fcfg,
    )