# core/env.py
import numpy as np
from .config import FieldCfg

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))  # ring metric
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))

def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)
    base = float(cfg.baseline)
    scale = float(cfg.amp_scale)

    for s in cfg.sources:
        if s.get("kind", "moving_peak") != "moving_peak":
            continue
        amp   = scale * float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.0)) * X   # cells per frame
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X

        pos = float(start)
        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X

    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)

    E += base
    np.maximum(E, 0.0, out=E)
    return E