from __future__ import annotations
import math, dataclasses
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Defaults exposed for the UI
# -----------------------------------------------------------------------------
default_config = dict(
    frames=1600,
    space=192,
    seed=0,
    # simple environment controls
    env_E0=2.0,     # base free energy
    env_var=0.5,    # variance of local fluctuations
    # growth / pruning knobs
    grow_budget=10,
    prune_tau=800,
)

# -----------------------------------------------------------------------------
# Minimal connection representation
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class Conn:
    kind: str        # "SENSE" | "INTERNAL" | "MOTOR"
    ell: float       # spatial/event scale
    h: float         # threshold-like internal scalar
    age: int = 0

# -----------------------------------------------------------------------------
# Simple environment & engine (toy) to keep the app demonstrable
# -----------------------------------------------------------------------------
class Engine:
    def __init__(self, cfg: dict):
        self.cfg = dict(default_config)
        self.cfg.update(cfg or {})
        self.rng = np.random.default_rng(self.cfg.get("seed", 0))

        # start with a small random set of connections, allow all three kinds
        self.conns: list[Conn] = []
        for k in ["SENSE", "INTERNAL", "MOTOR"]:
            for _ in range(3):
                self.conns.append(
                    Conn(kind=k, ell=float(self.rng.uniform(0.1, 1.0)), h=float(self.rng.uniform(-0.1, 0.1)))
                )

        # running state
        self.t = 0
        self.E = float(self.cfg["env_E0"])
        self.S = 0.0

    # a tiny “environment” — fluctuating free energy & structure
    def environment_tick(self):
        E0  = float(self.cfg["env_E0"])
        var = float(self.cfg["env_var"])
        dE  = self.rng.normal(0.0, var * 0.05)
        self.E = max(0.0, E0 + dE + 0.5 * math.sin(0.005 * self.t))
        # toy “entropy”: increases with noise; decreases when internals grow
        self.S = max(0.0, var + 0.1 * math.sin(0.01 * self.t))

    # local harvest function per-connection (toy & bounded)
    def harvest(self, c: Conn) -> float:
        # SENSE harvest mostly from boundary E, MOTOR can funnel a bit more if E high,
        # INTERNAL harvests from structure (S) & interactions in a weak way
        if c.kind == "SENSE":
            return 0.2 * self.E * (1.0 / (1.0 + abs(c.h)))
        if c.kind == "MOTOR":
            # motors benefit more from high E but pay a small activity cost (handled below)
            return 0.3 * self.E * (1.0 / (1.0 + abs(c.h)))
        # INTERNAL: small baseline; increases as entropy (structure) exists
        return 0.05 * (self.E + self.S) * (1.0 / (1.0 + abs(c.h)))

    def activity_cost(self, c: Conn) -> float:
        base = 0.01
        if c.kind == "MOTOR":
            base = 0.03
        return base * (1.0 + c.ell)

    def maintenance_cost(self, c: Conn) -> float:
        return 0.005 * (1.0 + c.ell)

    def step(self):
        self.environment_tick()

        # per-conn net contribution
        contrib = []
        for c in self.conns:
            P = self.harvest(c)
            C = self.activity_cost(c) + self.maintenance_cost(c)
            net = P - C
            contrib.append(net)
            c.age += 1

            # plasticity (lightweight): small random walk toward reducing |h|
            if self.rng.random() < 0.1:
                c.h += float(self.rng.normal(0.0, 0.01)) * np.sign(-c.h)

        # very small global adaptation: if E is high, grow motors; if S is high, grow internals;
        # occasionally spawn sensors. This is just to make the demo lively.
        if (self.t % 40) == 0:
            budget = int(self.cfg.get("grow_budget", 10))
            for _ in range(budget):
                r = self.rng.random()
                if r < 0.4:   # prefer motors when E is available
                    self.conns.append(Conn("MOTOR", ell=float(self.rng.uniform(0.1, 1.0)), h=0.0))
                elif r < 0.8: # internals fairly often
                    self.conns.append(Conn("INTERNAL", ell=float(self.rng.uniform(0.1, 1.0)), h=0.0))
                else:         # sometimes sensors
                    self.conns.append(Conn("SENSE", ell=float(self.rng.uniform(0.1, 1.0)), h=0.0))

        # pruning: remove oldest low-contributors with small probability
        if (self.t % int(self.cfg.get("prune_tau", 800))) == 0 and self.t > 0:
            # sort by contribution ascending and drop a few
            idx = np.argsort(contrib)
            drop = min(5, len(self.conns) // 10)
            for i in idx[:drop]:
                # guard bounds
                j = max(0, min(len(self.conns) - 1, int(i)))
                self.conns.pop(j)

        # aggregate “system” free energy as a softplus of contributions to keep it finite
        E_free = float(np.log1p(np.exp(np.clip(np.sum(contrib), -50, 50))))
        self.t += 1

        # counts for the UI
        n_sense    = sum(1 for c in self.conns if c.kind == "SENSE")
        n_internal = sum(1 for c in self.conns if c.kind == "INTERNAL")
        n_motor    = sum(1 for c in self.conns if c.kind == "MOTOR")

        return dict(
            frame=self.t,
            E=E_free,
            S=self.S,
            n_sense=n_sense,
            n_internal=n_internal,
            n_motor=n_motor,
        )

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for i, c in enumerate(self.conns):
            rows.append(dict(index=i, type=c.kind, L=c.ell, T=0.0, delay_tau=0.0, alpha_tau=0.0, Age=c.age))
        return pd.DataFrame(rows)


# Optional factory (exported so direct imports work if you really want them)
def make_engine(cfg: dict) -> Engine:
    return Engine(cfg)