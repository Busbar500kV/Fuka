# sim_core.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np

# ---------------- cfg ----------------

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 2000
    noise_sigma: float = 0.005
    # list of dicts: {"kind":"moving_peak","amp":float,"speed":float,"width":float,"start":int}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.2, "speed": 0.02, "width": 6.0, "start": 20}
    ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 2000
    space: int = 64
    n_init: int = 9
    k_flux: float = 0.08
    k_motor: float = 1.5
    decay: float = 0.01
    diffuse: float = 0.08
    env: FieldCfg = field(default_factory=FieldCfg)

# ------------- env builder -------------

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))

def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)
    for s in cfg.sources:
        if s.get("kind", "moving_peak") != "moving_peak":
            continue
        amp   = float(s.get("amp",   1.0))
        speed = float(s.get("speed", 0.1)) * X
        width = float(s.get("width", 4.0))
        pos   = float(int(s.get("start", 0)) % X)
        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X
    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)
    np.maximum(E, 0.0, out=E)
    return E

# ------------- history -------------

@dataclass
class History:
    t: List[int] = field(default_factory=list)
    E_cell: List[float] = field(default_factory=list)
    E_env:  List[float] = field(default_factory=list)
    E_flux: List[float] = field(default_factory=list)

# ------------- engine -------------

class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = build_env(cfg.env, self.rng)  # (T, X_env)
        self.T   = cfg.frames
        self.X   = cfg.space
        self.S   = np.zeros((self.T, self.X), dtype=float)
        self.hist = History()

    def _resample_env_row(self, t: int) -> np.ndarray:
        row = self.env[t]
        if self.env.shape[1] == self.X:
            return row
        # periodic index mapping (no heavy interp; keeps speed & periodicity)
        idx = (np.arange(self.X) * self.env.shape[1] // self.X) % self.env.shape[1]
        return row[idx]

    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        e_row = self._resample_env_row(t)
        prev = self.S[t-1] if t > 0 else self.S[0]

        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        band = 3
        grad = np.maximum(e_row[:band] - S_decayed[:band], 0.0)
        pump = np.zeros_like(S_decayed)
        pump[:band] += cfg.k_flux * grad

        mot = np.zeros_like(S_decayed)
        mot[:band] += cfg.k_motor * self.rng.random(band)

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

    # Accept snapshot_every (ignored) for UI compatibility
    def run(self, progress_cb: Optional[Callable[[int], None]] = None,
            snapshot_every: Optional[int] = None, **kwargs) -> None:
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)

# ------------- public helpers -------------

def default_config() -> Dict:
    return asdict(Config())

def make_engine(cfg_dict: Dict) -> Engine:
    """Build Engine from a plain dict (keeps UI decoupled from dataclass ctor)."""
    env_dict = cfg_dict.get("env", {})
    fcfg = FieldCfg(
        length=int(env_dict.get("length", 512)),
        frames=int(env_dict.get("frames", cfg_dict.get("frames", 2000))),
        noise_sigma=float(env_dict.get("noise_sigma", 0.005)),
        sources=env_dict.get("sources", FieldCfg().sources),
    )
    ecfg = Config(
        seed=int(cfg_dict.get("seed", 0)),
        frames=int(cfg_dict.get("frames", 2000)),
        space=int(cfg_dict.get("space", 64)),
        n_init=int(cfg_dict.get("n_init", 9)),
        k_flux=float(cfg_dict.get("k_flux", 0.08)),
        k_motor=float(cfg_dict.get("k_motor", 1.5)),
        decay=float(cfg_dict.get("decay", 0.01)),
        diffuse=float(cfg_dict.get("diffuse", 0.08)),
        env=fcfg,
    )
    return Engine(ecfg)