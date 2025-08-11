# sim_core.py
# Minimal, self-contained simulation core with a streaming-capable Engine.
# - Frames are synced between UI cfg and env cfg to avoid index errors
# - Engine uses T_eff = min(sim_frames, env_frames) for extra safety
# - Default env has a constant source near the boundary + one moving peak
# - JSON-configurable sources

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
    frames: int = 5000
    noise_sigma: float = 0.01
    # Each source can be:
    #   {"kind": "constant", "amp": 1.0, "start": 12, "width": 8}
    #   {"kind": "moving_peak", "amp": 0.6, "speed": 0.08, "width": 6.0, "start": 140}
    #   {"kind": "stationary_peak", "amp": 0.8, "width": 5.0, "pos": 24}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "constant",        "amp": 1.0, "start": 8,  "width": 10},     # boundary free-energy tap
        {"kind": "moving_peak",     "amp": 0.6, "speed": 0.08, "width": 6.0, "start": 180},
    ])


@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64
    n_init: int = 9  # placeholder for future connection-based model
    # energy / dynamics knobs
    k_flux: float  = 0.05    # how strongly boundary pumps env→cell when gradient exists
    k_motor: float = 2.0     # random motor exploration along boundary band
    k_noise: float = 0.00    # extra band noise (Gaussian) for “twitch”
    decay: float   = 0.01    # substrate local decay
    diffuse: float = 0.05    # substrate diffusion strength
    band: int      = 3       # width of the boundary band that interacts with env
    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Helper: environment field
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Return a gaussian-shaped bump at position `pos` (wrap around)."""
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))  # ring distance
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """Create env field E[t, x] from sources."""
    T, X = int(cfg.frames), int(cfg.length)
    E = np.zeros((T, X), dtype=float)

    for s in (cfg.sources or []):
        kind = str(s.get("kind", "moving_peak")).lower()

        if kind == "constant":
            amp   = float(s.get("amp", 1.0))
            start = int(s.get("start", 0)) % X
            width = int(max(1, int(s.get("width", 8))))
            mask = np.zeros(X, dtype=float)
            idxs = np.arange(start, start + width) % X
            mask[idxs] = 1.0
            E += amp * mask   # same every frame

        elif kind == "moving_peak":
            amp   = float(s.get("amp", 1.0))
            speed = float(s.get("speed", 0.1)) * X  # speed in cells/frame
            width = float(s.get("width", 4.0))
            start = int(s.get("start", 0)) % X
            pos = float(start)
            for t in range(T):
                E[t] += _moving_peak(X, pos, amp, width)
                pos = (pos + speed) % X

        elif kind == "stationary_peak":
            amp   = float(s.get("amp", 1.0))
            width = float(s.get("width", 4.0))
            pos   = float(s.get("pos", 0.0)) % X
            bump = _moving_peak(X, pos, amp, width)
            E += bump  # duplicate per frame

        else:
            # unknown kind – ignore
            continue

    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)

    # Clamp to non-negative “free energy”
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
    """Time-step simulation with simple, local rules and streaming callback."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # env and substrate
        self.env = build_env(cfg.env, self.rng)  # shape (T_env, X_env)
        self.T   = int(cfg.frames)
        self.X   = int(cfg.space)

        # Use the minimum of sim frames and env frames for safety
        self.T_eff = min(self.T, self.env.shape[0])

        # substrate state S[t, x]
        self.S = np.zeros((self.T_eff, self.X), dtype=float)
        self.S[0] = 0.0

        # energies (scalars per-step)
        self.hist = History()

    def _resample_env_row(self, e_row: np.ndarray) -> np.ndarray:
        """Resample/tile env row to substrate space size by modular indexing (keeps periodicity)."""
        if e_row.size == self.X:
            return e_row
        idx = (np.arange(self.X) * e_row.size // self.X) % e_row.size
        return e_row[idx]

    def step(self, t: int) -> Tuple[float, float, float]:
        """Advance by one step; returns (E_cell, E_env, E_flux) at this t."""
        cfg = self.cfg
        # env row for this step
        e_row = self._resample_env_row(self.env[t])

        # previous substrate
        prev = self.S[t-1] if t > 0 else self.S[0]

        # local diffusion + decay
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # boundary pump: env at boundary x in [0:band) acts as “free-energy tap”
        band = int(max(1, cfg.band))
        grad = np.maximum(e_row[:band] - S_decayed[:band], 0.0)  # only when env > substrate
        pump = np.zeros_like(S_decayed)
        pump[:band] += cfg.k_flux * grad

        # motor exploration along boundary band: small random pushes + optional band noise
        mot = np.zeros_like(S_decayed)
        mot[:band] += cfg.k_motor * self.rng.random(band)
        if cfg.k_noise > 0.0:
            mot[:band] += cfg.k_noise * self.rng.normal(size=band)

        # update substrate
        cur = S_decayed + pump + mot
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
        for t in range(self.T_eff):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)


# =========================
# Public helpers
# =========================

def default_config() -> Dict:
    """Return a plain dict config (useful for Streamlit session_state)."""
    return asdict(Config())  # converts nested dataclasses to plain dicts

def run_sim(cfg_dict: Dict) -> Tuple[History, np.ndarray, np.ndarray]:
    """Convenience: build engine from dict and run.
       Ensures env.frames == frames to keep shapes aligned.
    """
    # Pull UI values with defaults
    frames_ui = int(cfg_dict.get("frames", 5000))
    space_ui  = int(cfg_dict.get("space", 64))

    # Mirror frames into env JSON to avoid mismatches
    env_cfg_dict = dict(cfg_dict.get("env", {}) or {})
    env_cfg_dict["frames"] = frames_ui

    cfg = Config(
        seed   = int(cfg_dict.get("seed", 0)),
        frames = frames_ui,
        space  = space_ui,
        n_init = int(cfg_dict.get("n_init", 9)),
        k_flux = float(cfg_dict.get("k_flux", 0.05)),
        k_motor= float(cfg_dict.get("k_motor", 2.0)),
        k_noise= float(cfg_dict.get("k_noise", 0.00)),
        decay  = float(cfg_dict.get("decay", 0.01)),
        diffuse= float(cfg_dict.get("diffuse", 0.05)),
        band   = int(cfg_dict.get("band", 3)),
        env    = FieldCfg(
            length     = int(env_cfg_dict.get("length", 512)),
            frames     = int(env_cfg_dict.get("frames", frames_ui)),  # synced
            noise_sigma= float(env_cfg_dict.get("noise_sigma", 0.01)),
            sources    = env_cfg_dict.get("sources", FieldCfg().sources),
        ),
    )

    eng = Engine(cfg)
    eng.run()
    return eng.hist, eng.env, eng.S