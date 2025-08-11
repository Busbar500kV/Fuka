# sim_core.py
# Minimal free‑energy gradient sim with a *moving boundary window*.
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


# =========================
# Config
# =========================

@dataclass
class FieldCfg:
    """Environment field (space-time) configuration."""
    length: int = 512
    frames: int = 5000
    noise_sigma: float = 0.01
    # Each source: {"kind": "moving_peak", "amp": float, "speed": float, "width": float, "start": int}
    # Default: one stationary source near the boundary and one faint moving source.
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0,  "width": 6.0, "start": 40},   # close & constant
        {"kind": "moving_peak", "amp": 0.4, "speed": 0.06, "width": 8.0, "start": 260},  # far & weak
    ])


@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64        # substrate cells (the “organism” width)
    # dynamics
    k_flux: float = 0.05   # env -> cell pump gain at the boundary band
    k_motor: float = 2.0   # *position motor* gain (how strongly offset moves toward gradients)
    k_noise: float = 0.0   # optional direct noise injection into the band (usually 0)
    decay: float = 0.01    # substrate decay
    diffuse: float = 0.15  # substrate diffusion
    band: int = 3          # how many boundary cells couple to the env
    # env config
    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Environment builder
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Gaussian bump centered at `pos` on a ring of size `space`."""
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    w2 = max(1e-6, width) ** 2
    return amp * np.exp(-(d * d) / (2.0 * w2))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)
    for s in cfg.sources:
        if s.get("kind", "moving_peak") != "moving_peak":
            continue
        amp   = float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.0)) * X  # cells/frame
        width = float(s.get("width", 6.0))
        start = int(s.get("start", 0)) % X
        pos = float(start)
        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X

    if cfg.noise_sigma > 0.0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)
    np.maximum(E, 0.0, out=E)  # clamp to non-negative
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
    o: List[float]      = field(default_factory=list)  # boundary offset trace


# =========================
# Engine
# =========================

class Engine:
    """Time-step simulation with local rules + a *moving env window* driven by a simple motor."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # env and substrate
        self.env = build_env(cfg.env, self.rng)  # (T, X_env)
        self.env_len = self.env.shape[1]
        self.T = cfg.frames
        self.X = cfg.space

        # substrate S[t, x]
        self.S = np.zeros((self.T, self.X), dtype=float)

        # boundary position (offset into env ring)
        self.o = 0.0

        self.hist = History()

    # ---- helpers ----
    def _sample_env_window(self, t: int, offset: float) -> np.ndarray:
        """Take a length-X window from env[t] starting at `offset` (wrap)."""
        s = int(np.floor(offset)) % self.env_len
        idx = (s + np.arange(self.X)) % self.env_len
        return self.env[t, idx]

    def _band_reward_at(self, t: int, offset: float, S_band: np.ndarray) -> float:
        """Reward of placing the boundary band at `offset` (env - substrate)."""
        band = self.cfg.band
        s = int(np.floor(offset)) % self.env_len
        j = (s + np.arange(band)) % self.env_len
        # larger is better: what extra energy exists in env vs current S
        return float(np.sum(self.env[t, j] - S_band))

    # ---- one step ----
    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        prev = self.S[t-1] if t > 0 else self.S[0]

        # diffusion + decay
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5 * (left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # choose env window using current offset
        e_row = self._sample_env_window(t, self.o)

        # boundary pump in first `band` cells
        band = cfg.band
        grad = np.maximum(e_row[:band] - S_decayed[:band], 0.0)
        pump = np.zeros_like(S_decayed)
        pump[:band] += cfg.k_flux * grad

        # (optional) tiny direct noise injection into band
        if cfg.k_noise > 0.0:
            pump[:band] += cfg.k_noise * self.rng.random(band)

        # update substrate
        cur = S_decayed + pump
        self.S[t] = cur

        # ---- motor: update offset `self.o` by hill-climbing the band reward ----
        # Evaluate reward if we shift the window +1 or -1 env cell.
        g_plus  = self._band_reward_at(t, self.o + 1.0, S_decayed[:band])
        g_minus = self._band_reward_at(t, self.o - 1.0, S_decayed[:band])
        grad_o  = g_plus - g_minus  # sign tells where band energy increases
        eta     = float(cfg.k_motor) / max(1.0, self.env_len)  # normalize by env size
        noise   = 0.1 * eta * self.rng.normal()
        self.o  = (self.o + eta * grad_o + noise) % self.env_len

        # bookkeeping
        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump))

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)
        self.hist.o.append(self.o)

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
    """Return a plain dict for easy use with Streamlit session_state."""
    return asdict(Config())

def run_sim(cfg_dict: Dict) -> Tuple[History, np.ndarray, np.ndarray]:
    """Convenience: build engine from dict and run."""
    cfg = Config(
        seed=cfg_dict.get("seed", 0),
        frames=cfg_dict.get("frames", 5000),
        space=cfg_dict.get("space", 64),
        k_flux=cfg_dict.get("k_flux", 0.05),
        k_motor=cfg_dict.get("k_motor", 2.0),
        k_noise=cfg_dict.get("k_noise", 0.0),
        decay=cfg_dict.get("decay", 0.01),
        diffuse=cfg_dict.get("diffuse", 0.15),
        band=cfg_dict.get("band", 3),
        env=FieldCfg(
            length=cfg_dict.get("env", {}).get("length", 512),
            frames=cfg_dict.get("env", {}).get("frames", cfg_dict.get("frames", 5000)),
            noise_sigma=cfg_dict.get("env", {}).get("noise_sigma", 0.01),
            sources=cfg_dict.get("env", {}).get("sources", FieldCfg().sources),
        ),
    )
    eng = Engine(cfg)
    eng.run()
    return eng.hist, eng.env, eng.S