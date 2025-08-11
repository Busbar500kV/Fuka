# sim_core.py
# Core free‑energy gradient sim with a learned temporal gate (sensing DoF)
# and a movable boundary driven by a local motor. Designed for live streaming.

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
    space: int = 64  # substrate width (cells)
    # energy / dynamics knobs
    k_flux: float = 0.08       # boundary flux strength (when gate is open)
    k_motor: float = 1.00      # motor exploration magnitude (dimensionless)
    motor_noise: float = 0.02  # jitter in motor command
    c_motor: float = 0.80      # energetic cost per unit motor work
    decay: float = 0.01        # substrate local decay
    diffuse: float = 0.15      # substrate diffusion strength
    band: int = 2              # boundary width (cells affected by gate & motor)

    # temporal gate (emergent sensing) params
    gate_win: int = 30         # half window K => kernel len = 2K+1
    eta: float = 0.02          # kernel learning rate
    ema_beta: float = 0.10     # EMA for reward baseline
    lam_l1: float = 0.10       # L1 shrinkage
    prune_thresh: float = 0.10 # magnitude below which weights are pruned to 0

    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Helper: environment field
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Return a gaussian-shaped bump at position `pos` (wrap around)."""
    x = np.arange(space, dtype=float)
    # shortest distance on a ring
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """Create env field E[t, x] from sources."""
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)

    for s in cfg.sources:
        kind  = s.get("kind", "moving_peak")
        amp   = float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.0)) * X  # speed in cells/frame
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
    reward: List[float]  = field(default_factory=list)


# =========================
# Engine
# =========================

class Engine:
    """Time-step simulation with local rules, temporal gate, and streaming callback."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # env and substrate
        self.env = build_env(cfg.env, self.rng)  # shape (T, X_env)
        self.T = cfg.frames
        self.X = cfg.space

        # substrate state S[t, x]
        self.S = np.zeros((self.T, self.X), dtype=float)
        self.S[0] = 0.0

        # boundary (movable) index; 0 means left edge, we keep the band inside [0, X)
        self.bound = 0

        # gate kernel
        K = cfg.gate_win
        self.K = K
        self.w = np.zeros(2*K + 1, dtype=float)
        self.ema_baseline = 0.0

        # circular buffer of boundary samples for gate
        self._buf = np.zeros(2*K + 1, dtype=float)
        self._buf_idx = 0

        # live kernel history (downsampled; filled by run())
        self.w_hist_times: List[int] = []
        self.w_hist: List[np.ndarray] = []

        # energies (scalars per-step)
        self.hist = History()

    # ---- gate helpers ----
    def _buf_push(self, v: float):
        self._buf[self._buf_idx] = v
        self._buf_idx = (self._buf_idx + 1) % self._buf.size

    def _buf_read(self) -> np.ndarray:
        # return in chronological order (oldest .. newest)
        i = self._buf_idx
        return np.concatenate([self._buf[i:], self._buf[:i]])

    # ---- one step ----
    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg

        # env row for this step; if user set space!=env.length, we resample by modulo
        e_row_full = self.env[t]
        if self.env.shape[1] != self.X:
            idx = (np.arange(self.X) * self.env.shape[1] // self.X) % self.env.shape[1]
            e_row = e_row_full[idx]
        else:
            e_row = e_row_full

        # sample env at the current boundary cell for gate buffer
        boundary_cell = self.bound % self.X
        self._buf_push(float(e_row[boundary_cell]))

        # previous substrate
        prev = self.S[t-1] if t > 0 else self.S[0]

        # local diffusion + decay
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # ---- temporal gate (logistic) ----
        hist_window = self._buf_read()  # len = 2K+1
        # center-align: newest sample is at index 2K (end)
        # simple mean baseline to de-bias input to gate
        x_gate = hist_window - np.mean(hist_window)
        z = float(np.dot(self.w, x_gate))
        gate = 1.0 / (1.0 + np.exp(-z))  # in (0,1)

        # ---- boundary pump & motor ----
        b0 = self.bound % self.X
        band_idx = (b0 + np.arange(cfg.band)) % self.X

        # flux only when env > substrate in band
        grad = np.maximum(e_row[band_idx] - S_decayed[band_idx], 0.0)
        pump_band = cfg.k_flux * gate * grad

        pump = np.zeros_like(S_decayed)
        pump[band_idx] += pump_band

        # motor tries to shift the boundary in direction of strongest local gradient
        g_left  = float(np.sum(np.maximum(e_row[(b0-1) % self.X] - S_decayed[(b0-1) % self.X], 0.0)))
        g_right = float(np.sum(np.maximum(e_row[(b0+cfg.band) % self.X] - S_decayed[(b0+cfg.band) % self.X], 0.0)))
        dir_pref = np.sign(g_right - g_left)  # +1 move right, -1 left, 0 none

        motor_cmd = cfg.k_motor * dir_pref + cfg.motor_noise * self.rng.normal()
        move = int(np.clip(round(motor_cmd), -1, 1))
        work = abs(motor_cmd)
        self.bound = (self.bound + move) % self.X

        # update substrate
        cur = S_decayed + pump
        self.S[t] = cur

        # bookkeeping
        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump_band))

        # reward: flux revenue minus motor cost
        r = E_flux - cfg.c_motor * work
        # gate update: REINFORCE‑style baseline
        self.ema_baseline = (1.0 - cfg.ema_beta) * self.ema_baseline + cfg.ema_beta * r
        adv = r - self.ema_baseline
        grad_w = adv * x_gate  # ∂logπ/∂w ≈ x_gate for logistic gate
        self.w += cfg.eta * grad_w

        # L1 shrink + prune
        self.w = np.sign(self.w) * np.maximum(0.0, np.abs(self.w) - cfg.lam_l1 * cfg.eta)
        self.w[np.abs(self.w) < cfg.prune_thresh] = 0.0

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)
        self.hist.reward.append(r)

        return E_cell, E_env, E_flux

    def run(self, progress_cb: Optional[Callable[[int], None]] = None,
            snapshot_every: int = 100):
        for t in range(self.T):
            self.step(t)
            if (t % snapshot_every) == 0:
                # store kernel snapshot
                self.w_hist_times.append(t)
                self.w_hist.append(self.w.copy())
            if progress_cb is not None:
                progress_cb(t)


# =========================
# Public helpers
# =========================

def default_config() -> Dict:
    """Return a plain dict config (useful for Streamlit session_state)."""
    return asdict(Config())


def make_engine(cfg_dict: Dict) -> Engine:
    """Make an Engine from a plain dict."""
    cfg = Config(
        seed=cfg_dict.get("seed", 0),
        frames=int(cfg_dict.get("frames", 5000)),
        space=int(cfg_dict.get("space", 64)),

        k_flux=float(cfg_dict.get("k_flux", 0.08)),
        k_motor=float(cfg_dict.get("k_motor", 1.0)),
        motor_noise=float(cfg_dict.get("motor_noise", 0.02)),
        c_motor=float(cfg_dict.get("c_motor", 0.80)),
        decay=float(cfg_dict.get("decay", 0.01)),
        diffuse=float(cfg_dict.get("diffuse", 0.15)),
        band=int(cfg_dict.get("band", 2)),

        gate_win=int(cfg_dict.get("gate_win", 30)),
        eta=float(cfg_dict.get("eta", 0.02)),
        ema_beta=float(cfg_dict.get("ema_beta", 0.10)),
        lam_l1=float(cfg_dict.get("lam_l1", 0.10)),
        prune_thresh=float(cfg_dict.get("prune_thresh", 0.10)),

        env=FieldCfg(
            length=int(cfg_dict.get("env", {}).get("length", 512)),
            frames=int(cfg_dict.get("env", {}).get("frames", cfg_dict.get("frames", 5000))),
            noise_sigma=float(cfg_dict.get("env", {}).get("noise_sigma", 0.01)),
            sources=cfg_dict.get("env", {}).get("sources", FieldCfg().sources),
        ),
    )
    return Engine(cfg)