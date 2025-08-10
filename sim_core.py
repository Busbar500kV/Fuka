"""
sim_core.py — minimal but functional engine + strong diagnostics.

This file exposes:
  - default_config (dict)
  - Engine (class)
  - make_engine(cfg) -> Engine

It also includes diag_info() so the Streamlit app can print
where this module was loaded from and what it exports.
"""

from __future__ import annotations

import os
import inspect
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Generator, Optional, Tuple

import numpy as np


# -----------------------
# Public default config
# -----------------------
default_config: Dict[str, object] = {
    "seed": 0,
    "frames": 1600,
    "space": 192,                # just a label here
    "env_free_energy": 1.0,      # baseline environment energy
    "dt": 1.0,                   # time step
    "harvest_gain": 0.12,        # how strongly “motors” (currently implicit) harvest from env
    "internal_gain": 0.06,       # internal redistribution gain
    "cost_activity": 0.02,       # cost of being active
    "cost_maintenance": 0.01,    # base maintenance cost
    "noise_env": 0.15,           # environmental noise
}


# -----------------------------------
# Diagnostic helpers for the UI
# -----------------------------------
def _short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def diag_info() -> Dict[str, object]:
    """Return details so the UI can show what this module actually is."""
    here = os.path.abspath(__file__)
    exported = sorted([n for n in globals().keys() if not n.startswith("_")])
    try:
        src = inspect.getsource(Engine)
    except Exception:
        src = ""
    try:
        src2 = inspect.getsource(make_engine)
    except Exception:
        src2 = ""
    return {
        "module_file": here,
        "exported_symbols": exported,
        "engine_exists": "Engine" in globals(),
        "make_engine_exists": "make_engine" in globals(),
        "default_config_exists": "default_config" in globals(),
        "engine_source_hash": _short_hash(src) if src else None,
        "factory_source_hash": _short_hash(src2) if src2 else None,
    }


# -----------------------------------
# Minimal connection record (optional)
# -----------------------------------
@dataclass
class Conn:
    kind: str
    L: float
    T: float
    active_frac: float = 0.0
    energy_contrib: float = 0.0
    age: int = 0


# -----------------------------------
# The simulation engine
# -----------------------------------
class Engine:
    """
    A simple, stable simulation loop so the app can run and we can
    confirm imports. You can replace the internals later with your
    full free-energy dynamics.
    """

    def __init__(self, cfg: Dict[str, object]):
        self.cfg = dict(default_config)
        self.cfg.update(cfg or {})

        self.rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        self.t: int = 0

        # A tiny set of “boundary” connections for the demo.
        self.conns: List[Conn] = [
            Conn("SENSE", L=0.95, T=0.0),
            Conn("SENSE", L=0.85, T=0.0),
            Conn("SENSE", L=1.02, T=0.0),
        ]

        # Timeseries
        self.time: List[int] = []
        self.E_total: List[float] = []
        self.E_env: List[float] = []
        self.P_harvest: List[float] = []
        self.Costs: List[float] = []

        # State
        self.energy_internal: float = 0.0
        self.last_snapshot: Dict[str, object] = {}

    # -----------------------------
    # one time step
    # -----------------------------
    def step(self) -> None:
        dt = float(self.cfg["dt"])
        env_mean = float(self.cfg["env_free_energy"])
        env_noise = float(self.cfg["noise_env"])

        # Environment free energy at boundary this step
        E_env_t = max(0.0, env_mean + env_noise * self.rng.normal())

        # Harvesting (coarse placeholder)
        h_gain = float(self.cfg["harvest_gain"])
        # Simple “boundary intake” proportional to boundary L and env level:
        boundary_intake = E_env_t * h_gain * sum(max(0.0, c.L) for c in self.conns) / len(self.conns)

        # Internal redistribution (lossy gain)
        i_gain = float(self.cfg["internal_gain"])
        internal_boost = i_gain * max(0.0, self.energy_internal)

        # Costs
        c_act = float(self.cfg["cost_activity"])
        c_mai = float(self.cfg["cost_maintenance"])
        # Activity cost increases with average activity fraction
        act_frac = 0.0
        if self.conns:
            act_frac = np.clip(np.mean([c.active_frac for c in self.conns]), 0.0, 1.0)
        cost_activity = c_act * act_frac
        cost_maint = c_mai

        net = (boundary_intake + internal_boost) - (cost_activity + cost_maint)
        self.energy_internal += net * dt

        # Update “activity” for each connection in a tiny way
        for c in self.conns:
            # toy activity: more env energy => more active
            c.active_frac = float(np.clip(0.5 * E_env_t, 0.0, 1.0))
            c.energy_contrib = boundary_intake / max(1, len(self.conns))
            c.age += 1

        # Bookkeeping
        self.time.append(self.t)
        self.E_total.append(self.energy_internal)
        self.E_env.append(E_env_t)
        self.P_harvest.append(boundary_intake)
        self.Costs.append(cost_activity + cost_maint)
        self.t += 1

        # snapshot for the UI
        self.last_snapshot = {
            "t": self.t,
            "E_internal": self.energy_internal,
            "E_env": E_env_t,
            "harvest": boundary_intake,
            "costs": cost_activity + cost_maint,
            "active_frac": act_frac,
        }

    # -----------------------------
    # multiple steps with yielding
    # -----------------------------
    def run(self, steps: int, yield_every: int = 20) -> Generator[Dict[str, object], None, None]:
        for _ in range(steps):
            self.step()
            if yield_every > 0 and (self.t % yield_every == 0):
                yield self.last_snapshot

    # -----------------------------
    # resets
    # -----------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.cfg["seed"] = int(seed)
        self.__init__(self.cfg)

    # -----------------------------
    # simple dataframe-like dict
    # -----------------------------
    def connections_table(self) -> List[Dict[str, object]]:
        rows = []
        for idx, c in enumerate(self.conns):
            rows.append({
                "index": idx,
                "type": c.kind,
                "L": c.L,
                "T": c.T,
                "Active_Frac": c.active_frac,
                "Contribution_Energy": c.energy_contrib,
                "Age": c.age,
            })
        return rows

    # -----------------------------
    # tiny summary
    # -----------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "E_internal": self.energy_internal,
            "last_env": self.E_env[-1] if self.E_env else None,
            "num_conns": len(self.conns),
        }


# ---------------
# Public factory
# ---------------
def make_engine(cfg: Dict[str, object]):
    return Engine(cfg)


__all__ = ["default_config", "Engine", "make_engine", "diag_info"]