# sim_core.py
# Minimal, streaming-friendly core for the free-energy gradient simulation.

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
    length: int = 512            # env cells along boundary ring
    frames: int = 5000
    noise_sigma: float = 0.005
    # Each source: {"kind": "moving_peak", "amp": float, "speed": float, "width": float, "start": int}
    # speed is in fractions of ring per frame (we multiply by length)
    sources: List[Dict] = field(default_factory=lambda: [
        # A constant peak near boundary index 15 (close to x=0), no movement.
        {"kind": "moving_peak", "amp": 2.0, "speed": 0.0, "width": 6.0, "start": 15}
    ])


@dataclass
class Config:
    seed: int = 0
    frames: int = 5000           # substrate timeline
    space: int = 64              # substrate spatial cells
    # dynamics knobs
    k_flux: float = 0.12         # env->substrate pump at boundary band
    k_motor: float = 0.8         # random “motor” pokes in the band
    decay: float = 0.01          # local decay
    diffuse: float = 0.08        # local diffusion
    band: int = 3                # width of boundary band (cells from x=0)
    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Environment builder
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Gaussian-shaped bump at position `pos` (ring topology)."""
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    w = max(1e-6, float(width))
    return float(amp) * np.exp(-(d**2) / (2.0 * (w**2)))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """Create env field E[t, x] from sources."""
    T, X = int(cfg.frames), int(cfg.length)
    E = np.zeros((T, X), dtype=float)

    for s in cfg.sources:
        kind  = s.get("kind", "moving_peak")
        amp   = float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.0)) * X  # cells/frame around the ring
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X

        if kind != "moving_peak":
            continue

        pos = float(start)
        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X

    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, float(cfg.noise_sigma), size=E.shape)

    np.maximum(E, 0.0, out=E)  # free energy is non-negative
    return E


# =========================
# History buffers (for plots)
# =========================

@dataclass
class History:
    t: List[int] = field(default_factory=list)
    E_cell: List[float] = field(default_factory=list)
    E_env:  List[float] = field(default_factory=list)
    E_flux: List[float] = field(default_factory=list)


# =========================
# Engine (streaming-friendly)
# =========================

class Engine:
    """Time-step simulation with local rules and per-step streaming."""
    def __init__(self, cfg: Config):
        self.reset(cfg)

    def reset(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Build / resize env to match env.frames
        self.env = build_env(cfg.env, self.rng)          # shape (T_env, X_env)
        self.T = int(cfg.frames)
        self.X = int(cfg.space)
        # Substrate timeline
        self.S = np.zeros((self.T, self.X), dtype=float)
        self.S[0] = 0.0
        # Clear history
        self.hist = History()
        # Internal cursor
        self.t = 0

    def _env_row_resampled(self, t: int) -> np.ndarray:
        """Return env row at time t, resampled to substrate width if needed."""
        # Loop env time if frames mismatch so the app can run longer
        t_env = t % self.env.shape[0]
        row = self.env[t_env]
        if row.shape[0] == self.X:
            return row
        # simple periodic resample by index mapping (keeps structure and is fast)
        idx = (np.arange(self.X) * row.shape[0] // self.X) % row.shape[0]
        return row[idx]

    def step(self) -> Tuple[int, float, float, float]:
        """Advance one frame. Returns (t, E_cell, E_env, E_flux)."""
        if self.t >= self.T:
            # extend timeline if someone keeps running; grow S by chunks
            grow = max(256, self.T // 4)
            S_new = np.zeros((self.T + grow, self.X), dtype=float)
            S_new[:self.T] = self.S
            self.S = S_new
            self.T += grow

        cfg = self.cfg
        t = self.t
        prev = self.S[t-1] if t > 0 else self.S[0]

        # env row (resampled to substrate width)
        e_row = self._env_row_resampled(t)

        # diffusion + decay (ring)
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5 * (left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # boundary band pump (env -> substrate)
        b = int(max(1, cfg.band))
        grad = np.maximum(e_row[:b] - S_decayed[:b], 0.0)
        pump = np.zeros_like(S_decayed)
        pump[:b] += cfg.k_flux * grad

        # “motor” stochastic pokes in the band (exploration)
        mot = np.zeros_like(S_decayed)
        mot[:b] += cfg.k_motor * self.rng.random(b)

        # update state
        cur = S_decayed + pump + mot
        self.S[t] = cur

        # energy bookkeeping
        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump))

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)

        self.t += 1
        return t, E_cell, E_env, E_flux

    def run_chunk(self, n_steps: int, progress_cb: Optional[Callable[[int], None]] = None):
        """Run n_steps frames (for live streaming)."""
        end = self.t + int(n_steps)
        while self.t < end:
            t, *_ = self.step()
            if progress_cb is not None:
                progress_cb(t)

    # Legacy batch helper (runs full current T timeline)
    def run(self, progress_cb: Optional[Callable[[int], None]] = None):
        while self.t < self.T:
            t, *_ = self.step()
            if progress_cb is not None:
                progress_cb(t)


# =========================
# Public helpers
# =========================

def default_config() -> Dict:
    """Return a plain dict config for Streamlit session_state."""
    return asdict(Config())

def make_engine(cfg_dict: Dict) -> Engine:
    """Build an Engine from a plain dict."""
    cfg = Config(
        seed=int(cfg_dict.get("seed", 0)),
        frames=int(cfg_dict.get("frames", 5000)),
        space=int(cfg_dict.get("space", 64)),
        k_flux=float(cfg_dict.get("k_flux", 0.12)),
        k_motor=float(cfg_dict.get("k_motor", 0.8)),
        decay=float(cfg_dict.get("decay", 0.01)),
        diffuse=float(cfg_dict.get("diffuse", 0.08)),
        band=int(cfg_dict.get("band", 3)),
        env=FieldCfg(
            length=int(cfg_dict.get("env", {}).get("length", 512)),
            frames=int(cfg_dict.get("env", {}).get("frames", cfg_dict.get("frames", 5000))),
            noise_sigma=float(cfg_dict.get("env", {}).get("noise_sigma", 0.005)),
            sources=cfg_dict.get("env", {}).get("sources", FieldCfg().sources),
        ),
    )
    return Engine(cfg)

def run_sim(cfg_dict: Dict) -> Tuple[History, np.ndarray, np.ndarray]:
    """Batch run, mainly for non-live usage."""
    eng = make_engine(cfg_dict)
    eng.run()
    return eng.hist, eng.env, eng.S