# sim_core.py
# Stable API: Engine, make_engine, default_config

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


# =========================
# Environment config
# =========================

@dataclass
class FieldCfg:
    """Environment (space-time) configuration."""
    length: int = 512          # number of environment cells
    frames: int = 5000         # total frames
    noise_sigma: float = 0.00  # additive Gaussian noise
    # Supported kinds:
    #  - "constant_uniform": {"kind": "...", "amp": 0.0}
    #  - "constant_patch":   {"kind": "...", "amp": 1.0, "center": 15, "width": 8}
    #  - "moving_peak":      {"kind": "...", "amp": 1.0, "speed": 0.10, "width": 4.0, "start": 24}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "constant_patch", "amp": 1.0, "center": 15, "width": 10},
    ])


@dataclass
class Config:
    seed: int = 0
    frames: int = 3000
    space: int = 64
    # Dynamics knobs
    k_flux: float  = 0.08
    k_motor: float = 0.40
    k_noise: float = 0.02
    decay: float   = 0.01
    diffuse: float = 0.15
    band: int      = 3
    env: FieldCfg  = field(default_factory=FieldCfg)


# =========================
# Environment builders
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))  # ring distance
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))

def _constant_patch(space: int, center: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - center), space - np.abs(x - center))
    mask = (d <= width/2.0).astype(float)
    return amp * mask

def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """Make E[t, x] by summing all sources."""
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)

    for s in cfg.sources:
        kind = s.get("kind", "moving_peak")
        if kind == "moving_peak":
            amp   = float(s.get("amp", 1.0))
            speed = float(s.get("speed", 0.1)) * X  # cells per frame
            width = float(s.get("width", 4.0))
            start = float(s.get("start", 0.0)) % X
            pos = start
            for t in range(T):
                E[t] += _moving_peak(X, pos, amp, width)
                pos = (pos + speed) % X

        elif kind == "constant_patch":
            amp    = float(s.get("amp", 1.0))
            center = float(s.get("center", 15.0)) % X
            width  = float(s.get("width", 10.0))
            patch = _constant_patch(X, center, amp, width)
            for t in range(T):
                E[t] += patch

        elif kind == "constant_uniform":
            amp = float(s.get("amp", 0.0))
            E += amp

        else:
            continue

    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)

    np.maximum(E, 0.0, out=E)
    return E


# =========================
# Running history
# =========================

@dataclass
class History:
    t: List[int] = field(default_factory=list)
    E_cell: List[float] = field(default_factory=list)
    E_env:  List[float] = field(default_factory=list)
    E_flux: List[float] = field(default_factory=list)


# =========================
# Engine
# =========================

class Engine:
    """Simple 1D substrate coupled to a 1D environment with a boundary band."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = build_env(cfg.env, self.rng)   # (T_env, X_env)
        self.T = cfg.frames
        self.X = cfg.space
        self.S = np.zeros((self.T, self.X), dtype=float)
        self.S[0] = 0.0
        self.hist = History()

    def _env_row_resampled(self, t: int) -> np.ndarray:
        row = self.env[min(t, self.env.shape[0]-1)]
        if self.env.shape[1] == self.X:
            return row
        idx = (np.arange(self.X) * self.env.shape[1] // self.X) % self.env.shape[1]
        return row[idx]

    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        e_row = self._env_row_resampled(t)
        prev = self.S[t-1] if t > 0 else self.S[0]

        # local diffusion + decay
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # boundary pumps on a band near x=0
        b = int(max(1, cfg.band))
        band_slice = slice(0, b)

        # flux pump
        grad = np.maximum(e_row[band_slice] - S_decayed[band_slice], 0.0)
        pump = np.zeros_like(S_decayed)
        pump[band_slice] += cfg.k_flux * grad

        # motor exploration
        mot = np.zeros_like(S_decayed)
        mot[band_slice] += cfg.k_motor * self.rng.random(b)

        # tiny direct noise (symmetry breaking)
        noi = np.zeros_like(S_decayed)
        if cfg.k_noise > 0:
            noi[band_slice] += cfg.k_noise * (self.rng.random(b) - 0.5)

        cur = S_decayed + pump + mot + noi
        self.S[t] = cur

        # bookkeeping
        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump))

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)
        return E_cell, E_env, E_flux

    def run(self, progress_cb: Optional[Callable[[int], None]] = None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)


# =========================
# Public helpers
# =========================

def default_config() -> Dict:
    """Plain-dict defaults for easy editing in Streamlit."""
    return asdict(Config())

def make_engine(cfg_dict: Dict) -> Engine:
    """Create an Engine from a plain dict (robust to missing keys)."""
    env_dict = cfg_dict.get("env", {})
    env_cfg = FieldCfg(
        length      = int(env_dict.get("length", 512)),
        frames      = int(env_dict.get("frames", cfg_dict.get("frames", 3000))),
        noise_sigma = float(env_dict.get("noise_sigma", 0.0)),
        sources     = env_dict.get("sources", FieldCfg().sources),
    )
    cfg = Config(
        seed    = int(cfg_dict.get("seed", 0)),
        frames  = int(cfg_dict.get("frames", 3000)),
        space   = int(cfg_dict.get("space", 64)),
        k_flux  = float(cfg_dict.get("k_flux", 0.08)),
        k_motor = float(cfg_dict.get("k_motor", 0.40)),
        k_noise = float(cfg_dict.get("k_noise", 0.02)),
        decay   = float(cfg_dict.get("decay", 0.01)),
        diffuse = float(cfg_dict.get("diffuse", 0.15)),
        band    = int(cfg_dict.get("band", 3)),
        env     = env_cfg,
    )
    return Engine(cfg)