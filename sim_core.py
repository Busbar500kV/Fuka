# sim_core.py
# Minimal simulation core + simple “engine” to be driven by Streamlit.
# Focus here is correctness & clean API (make_engine / Engine.run / Engine.curves).

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


def default_config() -> Dict:
    return {
        "frames": 1600,     # total ticks
        "space": 192,       # spatial DOF
        "seed": 0,
        "env_level": 1.0,   # environment free-energy scale
        "motor_bias": 0.2,  # relative tendency toward “motor-like” action
        "internal_bias": 0.2,
        "sense_bias": 0.6,
        "chunk": 25,        # streaming redraw interval
        "sleep_ms": 0,      # artificial slowdown for streaming (0 = none)
    }


@dataclass
class Engine:
    cfg: Dict
    rng: np.random.Generator
    t: int
    curves: Dict[str, np.ndarray]
    info: Dict

    def __init__(self, cfg: Dict):
        self.cfg = dict(cfg)
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        T = int(self.cfg.get("frames", 1000))
        N = int(self.cfg.get("space", 128))

        # Curves we’ll expose to the UI
        self.curves = {
            "E_env": np.zeros(T, dtype=float),     # environment free energy
            "E_cell": np.zeros(T, dtype=float),    # internal/pool energy
            "E_flux": np.zeros(T, dtype=float),    # net flux into the cell
            "Sense": np.zeros(T, dtype=float),
            "Motor": np.zeros(T, dtype=float),
            "Internal": np.zeros(T, dtype=float),
        }
        self.info = {"space": N, "frames": T}
        self.t = 0

        # Internal “state”
        self._E = float(self.cfg.get("env_level", 1.0))  # environment level
        self._Cell = 0.1                                 # internal energy pool
        self._s = 0.0                                    # “sensor pressure”
        self._m = 0.0                                    # “motor pressure”
        self._i = 0.0                                    # “internal pressure”

        # Pre-build a smooth environment signal (no trig; smoothed noise)
        # start at env_level and wander via clipped random walk
        drift = self.rng.normal(0.0, 0.05, size=T)
        env = np.empty(T, dtype=float)
        level = float(self.cfg.get("env_level", 1.0))
        for k in range(T):
            level = level + 0.25 * drift[k]
            # squash into a positive band ~[0, 2] using softplus-like mapping
            # softplus(x) ~ log(1 + exp(x)); here we just clamp to keep finite
            level = float(np.clip(level, -3.0, 3.0))
            env[k] = math.log1p(math.exp(level))  # always > 0, smooth
        # normalize so mean ≈ env_level
        scale = float(self.cfg.get("env_level", 1.0)) / (env.mean() + 1e-9)
        env *= scale
        self._env_traj = env

    def _step_dynamics(self, t: int):
        """
        One tick of very simple, numerically-stable dynamics.
        We model:
          - E_env[t] from a smoothed random walk (precomputed)
          - Flux into cell depends on Motor (exploit/env-explore) + Sense (efficient capture)
          - Internal acts like catalytic structuring that increases efficiency over time
        All terms are small and bounded to avoid overflow.
        """
        # Read environment at t
        E_env = float(self._env_traj[t])

        # small random nudges to “pressures”
        self._m += 0.02 * (self.cfg["motor_bias"] - self._m) + 0.01 * self.rng.normal()
        self._s += 0.02 * (self.cfg["sense_bias"] - self._s) + 0.01 * self.rng.normal()
        self._i += 0.02 * (self.cfg["internal_bias"] - self._i) + 0.01 * self.rng.normal()

        # keep them bounded
        self._m = float(np.clip(self._m, 0.0, 1.0))
        self._s = float(np.clip(self._s, 0.0, 1.0))
        self._i = float(np.clip(self._i, 0.0, 1.0))

        # Efficiency rises with structure (internal) and sensing
        eff = 0.15 + 0.5 * self._s + 0.35 * self._i
        eff = float(np.clip(eff, 0.05, 0.98))

        # “Reach” rises with motor pressure but has diminishing returns
        reach = 0.1 + 0.8 * (1.0 - math.exp(-2.0 * self._m))
        reach = float(np.clip(reach, 0.05, 0.95))

        # Flux is product of available env, reach, and efficiency
        flux = E_env * reach * eff

        # Internal maintenance + mild leakage
        maintenance = 0.02 + 0.02 * (self._Cell)
        leak = 0.015 * self._Cell

        # Update cell energy (never below 0)
        dE = flux - (maintenance + leak)
        self._Cell = float(max(0.0, self._Cell + dE))

        # Save curves
        self.curves["E_env"][t] = E_env
        self.curves["E_cell"][t] = self._Cell
        self.curves["E_flux"][t] = dE
        self.curves["Sense"][t] = self._s
        self.curves["Motor"][t] = self._m
        self.curves["Internal"][t] = self._i

    def run(self, progress_cb: Optional[Callable[[int], None]] = None):
        T = self.info["frames"]
        sleep_ms = int(self.cfg.get("sleep_ms", 0))
        for t in range(T):
            self._step_dynamics(t)
            self.t = t
            if progress_cb is not None:
                progress_cb(t)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    # convenience for UI
    def curves_frame(self, upto: Optional[int] = None) -> pd.DataFrame:
        if upto is None:
            upto = self.t
        upto = max(0, min(int(upto), self.info["frames"] - 1))
        data = {
            "tick": np.arange(upto + 1),
            "E_env": self.curves["E_env"][:upto + 1],
            "E_cell": self.curves["E_cell"][:upto + 1],
            "E_flux": self.curves["E_flux"][:upto + 1],
            "Sense": self.curves["Sense"][:upto + 1],
            "Motor": self.curves["Motor"][:upto + 1],
            "Internal": self.curves["Internal"][:upto + 1],
        }
        return pd.DataFrame(data)


def make_engine(cfg: Dict) -> Engine:
    """Factory kept separate so the app can import it reliably."""
    return Engine(cfg)