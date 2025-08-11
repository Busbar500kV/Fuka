# sim_core.py — Sliding-gate motor with adaptive intake kernel (emergent sensor)

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
    # Each source: {"kind": "moving_peak", "amp": float, "speed": float, "width": float, "start": int}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.00, "width": 4.0, "start": 340},
    ])


@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64

    # substrate dynamics
    decay: float = 0.01
    diffuse: float = 0.15

    # boundary band
    band: int = 3               # boundary width (cells at x=[0..band-1])

    # motor / gate movement
    k_motor: float = 0.40       # motor command gain
    motor_noise: float = 0.02   # small motor jitter
    c_motor: float = 0.02       # motor work cost (energy debit per |u|)

    # intake (flux) from environment
    k_flux: float = 0.08        # how strongly intake adds to boundary
    # adaptive kernel ("sensor") over an env window around the gate
    gate_win: int = 8           # window half-width K -> size=2K+1
    eta: float = 0.02           # learning rate
    ema_beta: float = 0.10      # EMA speed for local prediction baseline
    lam_l1: float = 0.001       # L1 shrinkage
    prune_thresh: float = 1e-3  # small weight prune

    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Environment builders
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Gaussian bump at circular position `pos`."""
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))

def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """Create env field E[t, x] from sources (periodic in space)."""
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)
    for s in cfg.sources:
        if s.get("kind", "moving_peak") != "moving_peak":
            continue
        amp   = float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.0)) * X  # cells per frame
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X

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
    gate_pos: List[float] = field(default_factory=list)


# =========================
# Engine
# =========================

class Engine:
    """Time-step simulation with sliding gate and adaptive intake kernel."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # fields
        self.env = build_env(cfg.env, self.rng)  # (T, X_env)
        self.T   = cfg.frames
        self.X   = cfg.space
        self.Xe  = self.env.shape[1]

        # substrate
        self.S = np.zeros((self.T, self.X), dtype=float)

        # gate state & kernel
        self.gate_pos = float(0.0)                 # index in env space [0, Xe)
        K = cfg.gate_win
        self.w = np.abs(self.rng.normal(0.0, 1e-2, size=(2*K+1,)))  # small non-negative start
        s = np.sum(self.w)
        if s > 0: self.w /= s
        self.ema_win = np.zeros_like(self.w)        # EMA baseline of window values

        self.hist = History()

    # ---- helpers
    def _resample_e_row(self, t: int) -> np.ndarray:
        """Return env slice at time t, possibly resampled to X."""
        e_row = self.env[t]
        if self.Xe == self.X:
            return e_row
        idx = (np.arange(self.X) * self.Xe // self.X) % self.Xe
        return e_row[idx]

    def _take_env_window(self, e_row_full: np.ndarray) -> np.ndarray:
        """Local env window around the gate (size 2K+1) from full env (length Xe)."""
        K = self.cfg.gate_win
        c = int(np.floor(self.gate_pos)) % self.Xe
        idx = (np.arange(c-K, c+K+1) % self.Xe)
        return e_row_full[idx]

    # ---- core step
    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg

        # env rows
        e_row_full = self.env[t]              # for gate window (length Xe)
        e_row      = self._resample_e_row(t)  # for boundary gradient against substrate (length X)

        # previous substrate
        prev = self.S[t-1] if t > 0 else self.S[0]

        # local diffusion + decay on ring
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # ========== intake (emergent sensing)
        win = self._take_env_window(e_row_full)                   # size 2K+1
        # adaptive kernel dot the window
        intake_raw = float(np.dot(self.w, win))                   # scalar
        # compare to boundary level (mean over band)
        bmean = float(np.mean(S_decayed[:cfg.band]))
        intake = cfg.k_flux * max(intake_raw - bmean, 0.0)

        pump = np.zeros_like(S_decayed)
        if cfg.band > 0:
            pump[:cfg.band] += intake / cfg.band

        # ========== motor: move gate along env ring
        # crude directional bias: difference between right/left halves of the window
        K = cfg.gate_win
        bias = float(np.sum(win[K+1:]) - np.sum(win[:K])) / max(1.0, np.sum(win))
        u = cfg.k_motor * bias + cfg.motor_noise * self.rng.normal()
        self.gate_pos = (self.gate_pos + u) % self.Xe

        # motor energetic cost: subtract uniformly from boundary (can’t go below 0)
        debit = cfg.c_motor * abs(u)
        if debit > 0:
            debt = debit / max(1, cfg.band)
            S_decayed[:cfg.band] = np.maximum(0.0, S_decayed[:cfg.band] - debt)

        # ========== update substrate
        cur = S_decayed + pump
        self.S[t] = cur

        # ========== local plasticity (online, purely local to the boundary)
        # EMA baseline of the env window (predictable background)
        self.ema_win = (1.0 - cfg.ema_beta) * self.ema_win + cfg.ema_beta * win
        err = win - self.ema_win               # “surprise” drive
        a = float(np.mean(cur[:cfg.band]))     # boundary activity (proxy for harvested energy)
        self.w += cfg.eta * a * err            # Hebbian-like, driven by surprise
        # L1 shrink & prune
        self.w -= cfg.lam_l1 * np.sign(self.w)
        self.w[np.abs(self.w) < cfg.prune_thresh] = 0.0
        # keep kernel bounded (energy‑neutral scaling)
        s = float(np.sum(np.abs(self.w)))
        if s > 0:
            self.w /= s

        # bookkeeping
        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump[:cfg.band]))

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)
        self.hist.gate_pos.append(self.gate_pos)

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
    """Return a plain dict config (handy for Streamlit session_state)."""
    return asdict(Config())

def make_engine(cfg_dict: Dict) -> Engine:
    cfg = Config(
        seed=int(cfg_dict.get("seed", 0)),
        frames=int(cfg_dict.get("frames", 5000)),
        space=int(cfg_dict.get("space", 64)),
        decay=float(cfg_dict.get("decay", 0.01)),
        diffuse=float(cfg_dict.get("diffuse", 0.15)),
        band=int(cfg_dict.get("band", 3)),
        k_motor=float(cfg_dict.get("k_motor", 0.40)),
        motor_noise=float(cfg_dict.get("motor_noise", 0.02)),
        c_motor=float(cfg_dict.get("c_motor", 0.02)),
        k_flux=float(cfg_dict.get("k_flux", 0.08)),
        gate_win=int(cfg_dict.get("gate_win", 8)),
        eta=float(cfg_dict.get("eta", 0.02)),
        ema_beta=float(cfg_dict.get("ema_beta", 0.10)),
        lam_l1=float(cfg_dict.get("lam_l1", 0.001)),
        prune_thresh=float(cfg_dict.get("prune_thresh", 1e-3)),
        env=FieldCfg(
            length=int(cfg_dict.get("env", {}).get("length", 512)),
            frames=int(cfg_dict.get("env", {}).get("frames", cfg_dict.get("frames", 5000))),
            noise_sigma=float(cfg_dict.get("env", {}).get("noise_sigma", 0.01)),
            sources=cfg_dict.get("env", {}).get("sources", FieldCfg().sources),
        ),
    )
    return Engine(cfg)

def run_sim(cfg_dict: Dict) -> Tuple[History, np.ndarray, np.ndarray]:
    eng = make_engine(cfg_dict)
    eng.run()
    return eng.hist, eng.env, eng.S