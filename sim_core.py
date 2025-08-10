# sim_core.py
# FUKA — minimal live engine with editable space‑time signal and logging
# No trig; only softplus, linear moves, and Gaussian/RBF kernels.

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from typing import Dict, Any, Optional

# ---------- small numerics ----------
def softplus(x: np.ndarray | float) -> np.ndarray | float:
    # stable softplus, avoids overflow in exp
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    big = x > 20
    small = x < -20
    mid = (~big) & (~small)
    out[big] = x[big]
    out[small] = np.exp(x[small])
    out[mid] = np.log1p(np.exp(x[mid]))
    return out

_rng = np.random.default_rng

# ---------- Space‑Time field ----------
@dataclass
class FieldCfg:
    space: int = 192
    frames: int = 1600
    seed: int = 0
    # DoF for the signal (K moving Gaussian “events”)
    K: int = 5
    amp: float = 1.0
    width_min: float = 6.0
    width_max: float = 22.0
    speed_min: float = 0.4
    speed_max: float = 1.6
    # where the boundary lives (0 .. space-1)
    boundary_idx: int = 0
    # global offset (lets you “move” energy away/towards boundary)
    offset: int = 0
    # shape preset: "gauss", "ridge", "blob"
    preset: str = "gauss"

class SpaceTimeField:
    """
    Editable space‑time signal with K DoF (centers, widths, velocities, weights).
    All motion is linear-in-time; kernels are Gaussian (no trig).
    """
    def __init__(self, cfg: FieldCfg):
        self.cfg = cfg
        self.rng = _rng(cfg.seed)
        S, T = cfg.space, cfg.frames

        K = cfg.K
        self.c = self.rng.uniform(0, S-1, size=K)         # centers
        self.v = self.rng.uniform(cfg.speed_min, cfg.speed_max, size=K) * self.rng.choice([-1,1], size=K)
        self.w = self.rng.uniform(0.6, 1.4, size=K) * cfg.amp
        self.s = self.rng.uniform(cfg.width_min, cfg.width_max, size=K)  # std (space)
        if cfg.preset == "ridge":
            # longer narrow structures
            self.s *= 0.6
            self.w *= 1.2
        elif cfg.preset == "blob":
            self.s *= 1.6
            self.w *= 0.8

        # buffers (filled on demand)
        self.last_t = -1
        self.last_slice: Optional[np.ndarray] = None

    def slice(self, t: int) -> np.ndarray:
        """Return S-length signal at frame t."""
        if t == self.last_t and self.last_slice is not None:
            return self.last_slice
        S = self.cfg.space
        x = (np.arange(S) + self.cfg.offset) % S

        # linear motion (reflect at edges)
        ct = self.c + self.v * t
        # reflect with “bounce”
        m = 2*(S-1)
        ct = np.abs((ct % m) - (S-1))

        signal = np.zeros(S, dtype=float)
        for ci, si, wi in zip(ct, self.s, self.w):
            d = x - ci
            g = np.exp(-(d*d)/(2.0*si*si))
            signal += wi * g
        self.last_t = t
        self.last_slice = signal
        return signal

    def perturb(self, scale: float = 1.2):
        """Multiply weights by scale (quick ‘free energy up’)."""
        self.w *= float(scale)
        # invalidate cache
        self.last_t = -1
        self.last_slice = None

    def as_params(self) -> Dict[str, Any]:
        return dict(c=self.c.tolist(), v=self.v.tolist(), w=self.w.tolist(), s=self.s.tolist())

# ---------- Engine / roles ----------
@dataclass
class Config:
    # environment + run
    field: FieldCfg = FieldCfg()
    frames: int = 1600
    seed: int = 0
    redraw_every: int = 20

    # “roles” strengths (emergent; we just log the energies they harvest)
    sense_bias: float = 0.3
    motor_bias: float = 0.3
    internal_bias: float = 0.4

    # costs
    act_cost: float = 0.01
    leak: float = 0.001

def default_config() -> Dict[str, Any]:
    return asdict(Config())

class Engine:
    """
    Very small ‘cell’ that tries to keep a free‑energy gradient vs environment.
    We track three emergent channels: SENSE, MOTOR, INTERNAL (not hard-coded as types,
    only as energy collection mechanisms). No trig; all local, linear, or softplus.
    """
    def __init__(self, cfg: Dict[str, Any]):
        # unpack
        self.cfg = cfg
        self.frames = int(cfg.get("frames", 1600))
        self.rng = _rng(int(cfg.get("seed", 0)))

        # field
        fcfg = cfg.get("field", {})
        if isinstance(fcfg, dict):
            fcfg = FieldCfg(**fcfg)
        self.field = SpaceTimeField(fcfg)

        # state
        self.E_cell = 1.0     # internal energy pool
        self.history: Dict[str, list] = {
            "E_cell": [], "E_env": [], "E_flux": [],
            "SENSE": [], "MOTOR": [], "INTERNAL": [],
        }
        # persistent “model” of signal (RBF bank learned online)
        self.M_centers = np.linspace(0, fcfg.space-1, 16)
        self.M_width = 12.0
        self.M_weights = np.zeros_like(self.M_centers)

    # --- helper: cheap RBF projection, no trig
    def _rbf(self, x: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
        # returns [len(x), len(centers)] activations
        d = x[:, None] - centers[None, :]
        return np.exp(-(d*d) / (2.0*width*width))

    def _update_model(self, env_slice: np.ndarray):
        # one‑step Hebbian-ish update
        x = np.arange(env_slice.size)
        Phi = self._rbf(x, self.M_centers, self.M_width)  # [S, M]
        y = env_slice
        # local update
        grad = Phi.T @ y / (y.size + 1e-9)
        self.M_weights = 0.98*self.M_weights + 0.02*grad

    def _predict(self) -> np.ndarray:
        x = np.arange(self.field.cfg.space)
        Phi = self._rbf(x, self.M_centers, self.M_width)
        return Phi @ self.M_weights

    # --- single step ---
    def step(self, t: int):
        s_t = self.field.slice(t)                 # environment at boundary+interior
        env_energy = float(np.mean(s_t))          # scalar proxy

        # SENSE harvest: correlation gain when our model matches env
        pred = self._predict()
        match = 1.0 - np.mean(np.abs(s_t - pred)) / (np.mean(np.abs(s_t)) + 1e-9)
        sense_gain = self.cfg["sense_bias"] * softplus(3.0*match) * 0.2

        # MOTOR harvest: push/pull relative to boundary gradient
        b = self.field.cfg.boundary_idx
        bwin = s_t[max(0, b-2):min(s_t.size, b+3)]
        grad = float(np.max(bwin) - np.min(bwin))
        # small explore term (random “pokes”)
        explore = self.rng.normal(0, 0.03)
        motor_gain = self.cfg["motor_bias"] * softplus(grad + explore) * 0.15

        # INTERNAL harvest: maintain temporal slope (cheap memory)
        if len(self.history["E_env"]) == 0:
            slope = 0.0
        else:
            slope = env_energy - self.history["E_env"][-1]
        internal_gain = self.cfg["internal_bias"] * softplus(0.8*abs(slope)) * 0.08

        # costs
        activity = sense_gain + motor_gain + internal_gain
        cost = self.cfg["act_cost"] * activity + self.cfg["leak"] * self.E_cell

        # update energy pool
        E_flux = (sense_gain + motor_gain + internal_gain) - cost
        self.E_cell = max(0.0, self.E_cell + E_flux)

        # update world model after “sensing”
        self._update_model(s_t)

        # log
        self.history["E_cell"].append(self.E_cell)
        self.history["E_env"].append(env_energy)
        self.history["E_flux"].append(E_flux)
        self.history["SENSE"].append(sense_gain)
        self.history["MOTOR"].append(motor_gain)
        self.history["INTERNAL"].append(internal_gain)

    # --- public API ---
    def run(self, progress_cb=None):
        for t in range(self.frames):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)

    def get_series(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v, dtype=float) for k, v in self.history.items()}

    def get_env_frame(self, t: int) -> np.ndarray:
        t = int(np.clip(t, 0, self.frames-1))
        return self.field.slice(t)

    def perturb_env(self, scale: float = 1.2):
        self.field.perturb(scale)

    def snapshot(self) -> Dict[str, Any]:
        out = dict(cfg=self.cfg, field_params=self.field.as_params(), history=self.get_series())
        return out

# convenience for the UI
def run_sim(cfg: Dict[str, Any], cb=None) -> Dict[str, np.ndarray]:
    eng = Engine(cfg)
    eng.run(progress_cb=cb)
    return eng.get_series()