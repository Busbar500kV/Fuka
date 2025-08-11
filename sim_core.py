from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np

# =========================
# Config data classes
# =========================

@dataclass
class FieldCfg:
    """Environment field (space-time) configuration."""
    length: int = 512
    frames: int = 10000
    noise_sigma: float = 0.01
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 6.0, "start": 15},
    ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64
    n_init: int = 9
    k_flux: float = 0.05
    k_motor: float = 2.0
    k_noise: float = 0.0         # new: direct substrate noise in boundary
    decay: float = 0.01
    diffuse: float = 0.05
    band: int = 3                # new: width of boundary interaction zone
    env: FieldCfg = field(default_factory=FieldCfg)

# =========================
# Helper: environment field
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))

def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)
    for s in cfg.sources:
        kind  = s.get("kind", "moving_peak")
        amp   = float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.1)) * X
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X
        if kind != "moving_peak":
            continue
        pos = float(start)
        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X
    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)
    np.maximum(E, 0.0, out=E)
    return E

# =========================
# History buffers
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
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = build_env(cfg.env, self.rng)
        self.T = cfg.frames
        self.X = cfg.space
        self.S = np.zeros((self.T, self.X), dtype=float)
        self.S[0] = 0.0
        self.hist = History()

    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        e_row = self.env[t]
        if self.env.shape[1] != self.X:
            idx = (np.arange(self.X) * self.env.shape[1] // self.X) % self.env.shape[1]
            e_row = e_row[idx]

        prev = self.S[t-1] if t > 0 else self.S[0]

        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        pump = np.zeros_like(S_decayed)
        mot  = np.zeros_like(S_decayed)

        grad = np.maximum(e_row[:cfg.band] - S_decayed[:cfg.band], 0.0)
        pump[:cfg.band] += cfg.k_flux * grad
        mot[:cfg.band]  += cfg.k_motor * self.rng.random(cfg.band)

        # NEW: direct band noise
        if cfg.k_noise > 0:
            mot[:cfg.band] += cfg.k_noise * self.rng.normal(size=cfg.band)

        cur = S_decayed + pump + mot
        self.S[t] = cur

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
    return asdict(Config())

def run_sim(cfg_dict: Dict) -> Tuple[History, np.ndarray, np.ndarray]:
    cfg = Config(
        seed=cfg_dict.get("seed", 0),
        frames=cfg_dict.get("frames", 1600),
        space=cfg_dict.get("space", 64),
        n_init=cfg_dict.get("n_init", 9),
        k_flux=cfg_dict.get("k_flux", 0.05),
        k_motor=cfg_dict.get("k_motor", 2.0),
        k_noise=cfg_dict.get("k_noise", 0.0),
        decay=cfg_dict.get("decay", 0.01),
        diffuse=cfg_dict.get("diffuse", 0.05),
        band=cfg_dict.get("band", 3),
        env=FieldCfg(
            length=cfg_dict.get("env", {}).get("length", 512),
            frames=cfg_dict.get("env", {}).get("frames", cfg_dict.get("frames", 1600)),
            noise_sigma=cfg_dict.get("env", {}).get("noise_sigma", 0.01),
            sources=cfg_dict.get("env", {}).get("sources", FieldCfg().sources),
        ),
    )
    eng = Engine(cfg)
    eng.run()
    return eng.hist, eng.env, eng.S