# fuka_core.py
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd

def default_config() -> Dict:
    return {
        "frames": 1600,
        "space": 192,
        "seed": 0,
        "env_level": 1.0,
        "motor_bias": 0.2,
        "internal_bias": 0.2,
        "sense_bias": 0.6,
        "chunk": 25,
        "sleep_ms": 0,
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

        self.curves = {
            "E_env": np.zeros(T, dtype=float),
            "E_cell": np.zeros(T, dtype=float),
            "E_flux": np.zeros(T, dtype=float),
            "Sense": np.zeros(T, dtype=float),
            "Motor": np.zeros(T, dtype=float),
            "Internal": np.zeros(T, dtype=float),
        }
        self.info = {"frames": T}
        self.t = 0

        self._E = float(self.cfg.get("env_level", 1.0))
        self._Cell = 0.1
        self._s = 0.0
        self._m = 0.0
        self._i = 0.0

        drift = self.rng.normal(0.0, 0.05, size=T)
        env = np.empty(T, dtype=float)
        level = float(self.cfg.get("env_level", 1.0))
        for k in range(T):
            level = level + 0.25 * drift[k]
            level = float(np.clip(level, -3.0, 3.0))
            env[k] = math.log1p(math.exp(level))
        scale = float(self.cfg.get("env_level", 1.0)) / (env.mean() + 1e-9)
        env *= scale
        self._env_traj = env

    def _step_dynamics(self, t: int):
        E_env = float(self._env_traj[t])

        self._m += 0.02 * (self.cfg["motor_bias"] - self._m) + 0.01 * self.rng.normal()
        self._s += 0.02 * (self.cfg["sense_bias"] - self._s) + 0.01 * self.rng.normal()
        self._i += 0.02 * (self.cfg["internal_bias"] - self._i) + 0.01 * self.rng.normal()

        self._m = float(np.clip(self._m, 0.0, 1.0))
        self._s = float(np.clip(self._s, 0.0, 1.0))
        self._i = float(np.clip(self._i, 0.0, 1.0))

        eff = 0.15 + 0.5 * self._s + 0.35 * self._i
        eff = float(np.clip(eff, 0.05, 0.98))

        reach = 0.1 + 0.8 * (1.0 - math.exp(-2.0 * self._m))
        reach = float(np.clip(reach, 0.05, 0.95))

        flux = E_env * reach * eff

        maintenance = 0.02 + 0.02 * (self._Cell)
        leak = 0.015 * self._Cell

        dE = flux - (maintenance + leak)
        self._Cell = float(max(0.0, self._Cell + dE))

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
    return Engine(cfg)