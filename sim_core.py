# --- sim_core.py ---
from dataclasses import dataclass, asdict

@dataclass
class Config:
    frames:int = 200
    space:int = 64
    seed:int = 0

def default_config():
    return asdict(Config())

def run_sim(cfg:dict):
    # tiny dummy “simulation” so import always works
    # (you can replace with your real code later)
    import numpy as np
    rng = np.random.default_rng(cfg.get("seed", 0))
    T = cfg.get("frames", 200)
    X = cfg.get("space", 64)
    env = rng.random((X, T)) * 0.2
    subs = env.cumsum(axis=1) / np.maximum(1, np.arange(T)+1)
    energy = float(subs.sum())
    return {
        "env": env,      # (X,T)
        "subs": subs,    # (X,T)
        "energy": energy # scalar
    }