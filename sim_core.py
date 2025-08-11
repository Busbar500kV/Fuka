# sim_core.py
# Streaming-capable engine with conservative boundary harvest + advection motors.
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


# --------------------------
# Config data classes
# --------------------------

@dataclass
class FieldCfg:
    """Environment field (space–time) configuration."""
    length: int = 512          # E(t, x_env) spatial length
    frames: int = 5000         # total frames for E
    noise_sigma: float = 0.01  # white noise added to E(t, x_env)
    # Sources are simple local generators; you can edit these from the UI as JSON.
    # Example element:
    # {"kind":"moving_peak","amp":1.0,"speed":-0.02,"width":5.0,"start":340}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": -0.02, "width": 5.0, "start": 340},
    ])


@dataclass
class Config:
    seed: int = 1
    frames: int = 5000
    space: int = 64             # number of substrate cells (interior “body”)
    # Local physics
    k_flux: float = 0.08        # boundary harvest gain (env -> substrate)
    k_motor: float = 4.90       # motor intensity (controls advection velocity scale)
    k_noise: float = 0.02       # extra random band agitation (adds to velocity, not energy)
    decay: float = 0.01         # substrate decay (per-step fractional loss)
    diffuse: float = 0.15       # Fickian diffusion strength
    band: int = 3               # boundary band width (cells)
    # Cost coefficients
    c_motor: float = 0.02       # motor work cost per |v| * S (local)
    # Environment field
    env: FieldCfg = field(default_factory=FieldCfg)


# --------------------------
# Environment generator
# --------------------------

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    """Wrapped Gaussian bump centered at pos (0..space-1)."""
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    w2 = max(width, 1e-6)**2
    return amp * np.exp(-(d*d) / (2.0*w2))

def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)

    for s in cfg.sources:
        if s.get("kind", "moving_peak") != "moving_peak":
            continue
        amp   = float(s.get("amp",   1.0))
        speed = float(s.get("speed", 0.0)) * X  # in cells / frame (wrap)
        width = float(s.get("width", 4.0))
        pos   = float(int(s.get("start", 0)) % X)

        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X

    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)

    np.maximum(E, 0.0, out=E)  # “free energy” cannot be negative
    return E


# --------------------------
# History buffers
# --------------------------

@dataclass
class History:
    t: List[int]        = field(default_factory=list)
    E_cell: List[float] = field(default_factory=list)  # total substrate energy
    E_env:  List[float] = field(default_factory=list)  # total env energy (row sum)
    E_flux: List[float] = field(default_factory=list)  # harvested (env->sub) that step


# --------------------------
# Engine
# --------------------------

class Engine:
    """
    Local rules:
      - Boundary harvest (conservative): E_env loses exactly what substrate gains at band.
      - Diffusion (interior).
      - Motor advection: velocity field v acts on S via upwind flux; does NOT mint energy.
                         Cost for |v|*S is removed locally (motor work).
      - Decay.
    Everything is ring-topology along x.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Environment field E(t, x_env) and sizes
        self.env = build_env(cfg.env, self.rng)
        self.T_env, self.X_env = self.env.shape

        # Substrate field S(t, x_substrate). We keep all rows for convenient live heatmap.
        self.T = cfg.frames
        self.X = cfg.space
        self.S = np.zeros((self.T, self.X), dtype=float)

        # History
        self.hist = History()

        # Precompute resample mapping from env->substrate space (linear interp indices)
        self._env_x = np.linspace(0, self.X_env - 1, self.X, dtype=float)

    # ---------- helpers

    def _env_row_resampled(self, t: int) -> np.ndarray:
        """Resample E[t, :] onto substrate length using linear interpolation (periodic)."""
        row = self.env[min(t, self.T_env-1)]
        # periodic extension by wrapping indices
        i0 = np.floor(self._env_x).astype(int) % self.X_env
        i1 = (i0 + 1) % self.X_env
        w  = self._env_x - np.floor(self._env_x)
        return (1.0 - w) * row[i0] + w * row[i1]

    @staticmethod
    def _roll(a: np.ndarray, k: int) -> np.ndarray:
        return np.roll(a, k)

    # Upwind 1D conservative advection for scalar S with velocity v at cell centers.
    # dt = dx = 1.0 (Courant condition handled by clipping v below).
    def _advect(self, S: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Face velocities (centered to faces)
        v_r = 0.5 * (v + self._roll(v, -1))  # velocity at right face between i and i+1
        v_l = self._roll(v_r, 1)             # left face

        # Upwind fluxes
        S_up_r = np.where(v_r >= 0, S, self._roll(S, -1))
        S_up_l = np.where(v_l >= 0, self._roll(S, 1), S)

        F_r = v_r * S_up_r
        F_l = v_l * S_up_l

        return S - (F_r - F_l)  # dt=dx=1

    # ---------- step

    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg

        # Previous substrate
        prev = self.S[t-1] if t > 0 else self.S[0]

        # Resampled environment row (length = X)
        e_row = self._env_row_resampled(t)

        # 1) Diffusion
        left, right = self._roll(prev, 1), self._roll(prev, -1)
        S_diff = prev + cfg.diffuse * (0.5 * (left + right) - prev)

        # 2) Conservative boundary harvest (env -> substrate) on first `band` cells
        band = max(1, int(cfg.band))
        grad = np.maximum(e_row[:band] - S_diff[:band], 0.0)
        harvest = cfg.k_flux * grad                      # how much we can take
        # Limit by what's available in env (conservative)
        env_avail = np.maximum(e_row[:band], 0.0)
        take = np.minimum(harvest, env_avail)
        pump = np.zeros_like(S_diff); pump[:band] += take

        # Update a *shadow* env row so total env energy accounting stays correct for history.
        # (We don't mutate self.env; history uses sums after harvest.)
        e_after = e_row.copy()
        e_after[:band] -= take
        e_after = np.maximum(e_after, 0.0)

        # 3) Motors: produce a local velocity field v at the boundary band only.
        #           v is signed and has bounded magnitude for numerical stability.
        v = np.zeros_like(S_diff)
        # Local noisy motor signal in band
        raw = cfg.k_motor * (self.rng.random(band) - 0.5) + cfg.k_noise * self.rng.standard_normal(band)
        # Squash to safe Courant number (|v| <= 0.45)
        v[:band] = 0.45 * np.tanh(raw)

        # Apply advection using upwind scheme
        S_adv = self._advect(S_diff, v)

        # Motor energetic cost: remove locally, proportional to |v|*S
        cost = cfg.c_motor * np.abs(v) * S_adv
        S_after_motor = np.maximum(S_adv - cost, 0.0)

        # 4) Add boundary pump
        S_pumped = S_after_motor + pump

        # 5) Decay
        cur = (1.0 - cfg.decay) * S_pumped
        self.S[t] = cur

        # Bookkeeping (energy-like scalars)
        E_cell = float(np.sum(cur))
        E_env  = float(np.sum(e_after))         # total env energy after harvest at this row
        E_flux = float(np.sum(take))            # env -> substrate this step

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)

        return E_cell, E_env, E_flux

    # ---------- run

    def run(self, progress_cb: Optional[Callable[[int], None]] = None):
        for t in range(self.cfg.frames):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)


# --------------------------
# Public helpers (for Streamlit)
# --------------------------

def default_config() -> Dict:
    return asdict(Config())

def make_engine(cfg_dict: Dict) -> Engine:
    cfg = Config(
        seed   = int(cfg_dict.get("seed", 1)),
        frames = int(cfg_dict.get("frames", 5000)),
        space  = int(cfg_dict.get("space", 64)),
        k_flux = float(cfg_dict.get("k_flux", 0.08)),
        k_motor= float(cfg_dict.get("k_motor", 4.90)),
        k_noise= float(cfg_dict.get("k_noise", 0.02)),
        decay  = float(cfg_dict.get("decay", 0.01)),
        diffuse= float(cfg_dict.get("diffuse", 0.15)),
        band   = int(cfg_dict.get("band", 3)),
        c_motor= float(cfg_dict.get("c_motor", 0.02)),
        env    = FieldCfg(
            length     = int(cfg_dict.get("env", {}).get("length", 512)),
            frames     = int(cfg_dict.get("env", {}).get("frames", cfg_dict.get("frames", 5000))),
            noise_sigma= float(cfg_dict.get("env", {}).get("noise_sigma", 0.01)),
            sources    = cfg_dict.get("env", {}).get("sources", FieldCfg().sources),
        ),
    )
    return Engine(cfg)

def run_sim(cfg_dict: Dict) -> Tuple[History, np.ndarray, np.ndarray]:
    eng = make_engine(cfg_dict)
    eng.run()
    return eng.hist, eng.env, eng.S