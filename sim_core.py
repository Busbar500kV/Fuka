# sim_core.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

# ---------------- default config ----------------
def default_config():
    return dict(
        frames=1600,
        space=192,
        seed=0,
        grow_every=40,
        grow_attempts=3,
        log_every=40,
        env_energy=1.0,         # free energy at boundary (higher -> favors motors)
        env_persistence=0.95,   # environment inertia (0..0.999)
        env_variability=0.15,   # noise / event variability (0..1)
        save_drive_trace=True,
        step_chunk=5,           # frames advanced per UI loop tick
    )

# ---------------- simple connection model (placeholder) ----------------
@dataclass
class Conn:
    idx: int
    kind: str          # "SENSE"/"MOTOR"/"INTERNAL" (emergent; here just label)
    L: float
    T: float
    active_frac: float = 0.0
    contrib_energy: float = 0.0
    age: int = 0
    birth_frame: int = 0

def _grow_random_conns(rng, t, how_many):
    conns = []
    for k in range(how_many):
        kind = "SENSE"  # will “emerge” later; label isn’t used for logic here
        conns.append(Conn(
            idx=-1, kind=kind,
            L=float(rng.uniform(0.5, 1.1)),
            T=0.0, active_frac=0.0, contrib_energy=0.0, age=0, birth_frame=t
        ))
    return conns

# ---------------- Engine ----------------
class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg.copy()
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.t = 0
        self.S = np.zeros((cfg["space"],), dtype=np.float32)  # substrate line
        self.energy = 50.0                                    # global energy pool
        self.energy_series = []
        self.event_log = []
        self.drive_trace_head = []
        self.conns: list[Conn] = _grow_random_conns(self.rng, 0, 3)

        # environment buffers (for plotting + dynamics)
        self.env_last = np.zeros_like(self.S)
        self.env_stack = []   # keep short window for heatmap if needed

    # --- environment update ---
    def _env_step(self):
        p = self.cfg["env_persistence"]
        v = self.cfg["env_variability"]
        base = self.env_last * p
        # boundary free energy spikes
        bump = np.zeros_like(base)
        bump[0]  += self.cfg["env_energy"] * (0.6 + 0.4 * self.rng.random())
        bump[-1] += self.cfg["env_energy"] * (0.6 + 0.4 * self.rng.random())
        # sparse events inside
        if self.rng.random() < 0.15 + 0.35*v:
            x = self.rng.integers(0, base.size)
            bump[x] += (0.5 + v) * (0.5 + self.rng.random())
        env = np.maximum(0.0, base + bump)
        self.env_last = env
        return env

    # --- single frame physics (toy but stable) ---
    def _frame(self):
        env = self._env_step()

        # motors try to pump energy where env is strong near boundary
        pump_eff = min(1.0, 0.1 + 0.2*self.cfg["env_energy"])
        motor_gain = pump_eff * (env[0] + env[-1]) * 0.02

        # substrate relax/diffuse + driven by env
        self.S = 0.92*self.S + 0.06*np.roll(self.S, 1) + 0.06*np.roll(self.S, -1)
        self.S += 0.08*env

        # harvest from matching |dS/dt|-like activity (crude, but monotone)
        activity = np.abs(self.S - self.S.mean())
        harvest = float(activity.mean()) * (0.8 + 0.2*self.cfg["env_energy"])
        harvest += motor_gain

        # costs scale with number of connections and activity
        cost = 0.03*len(self.conns) + 0.15*float(activity.mean())

        dE = harvest - cost
        self.energy += dE
        self.energy_series.append(self.energy)

        # update conns stats a little
        for c in self.conns:
            c.age += 1
            c.active_frac = 0.98*c.active_frac + 0.02*float(activity.mean() > 0.02)
            c.contrib_energy += 0.5*dE/len(self.conns) if self.conns else 0.0

        # growth occasionally
        if (self.t % int(self.cfg["grow_every"])) == 0 and self.t > 0:
            new_conns = _grow_random_conns(self.rng, self.t, int(self.cfg["grow_attempts"]))
            # assign indices
            base = len(self.conns)
            for i, c in enumerate(new_conns):
                c.idx = base+i
            self.conns.extend(new_conns)
            self.event_log.append((self.t, "GROW", {"n": len(new_conns)}))

        # trace (head only to keep small)
        if len(self.drive_trace_head) < 400:
            self.drive_trace_head.append(dict(
                frame=self.t, activity=float(activity.mean()),
                harvest=harvest, cost=cost, dE=dE, energy=self.energy
            ))

        self.t += 1
        # keep short env stack buffer (last ~200 frames)
        if len(self.env_stack) >= 200:
            self.env_stack.pop(0)
        self.env_stack.append(env.copy())

    # public step
    def step(self, n=1):
        for _ in range(int(n)):
            self._frame()

    # current snapshot for UI
    def snapshot(self):
        table = [{
            "index": i,
            "type": c.kind,
            "L": c.L,
            "T": c.T,
            "Active_Frac": c.active_frac,
            "Contribution_Energy": c.contrib_energy,
            "Age": c.age,
            "birth_frame": c.birth_frame
        } for i, c in enumerate(self.conns)]
        return dict(
            t=self.t,
            energy_series=self.energy_series[-800:],  # last window
            env=np.array(self.env_stack).T if self.env_stack else None,
            substrate=self.S.copy(),
            conn_table=table,
            drive_trace_head=self.drive_trace_head,
            event_log_head=self.event_log[-200:],
        )

# convenience factory (used by the app)
def make_engine(cfg: dict):
    return Engine(cfg)