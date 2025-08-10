# sim_core.py
# Minimal, numerically safe engine for the Fuka free‑energy idea

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------------- default configuration ----------------
default_config = {
    "frames": 1600,          # history buffer
    "space": 192,            # spatial cells
    "seed": 0,

    # environment
    "env.energy": 2.0,       # total scale of free energy
    "env.boundary_bias": 0.6,# 0=deep interior, 1=near boundary
    "env.num_tracks": 4,     # moving energy packets
    "env.noise": 0.03,       # background noise level

    # substrate
    "subs.decay": 0.98,
    "subs.couple_env": 0.35,
    "subs.couple_conn": 0.25,

    # connections
    "conn.init_count": 8,
    "conn.max": 64,
    "conn.grow_every": 60,

    # economy (all dimensionless)
    "econ.cost.activation": 0.001,
    "econ.cost.maintenance": 0.0005,
    "econ.harvest.direct": 0.25,
    "econ.harvest.indirect": 0.25,
    "econ.harvest.motor": 0.25,

    # UI
    "ui.steps_per_tick": 5,
    "ui.refresh_ms": 300,
}

# ---------------- utility ----------------
rng_global = np.random.default_rng

def softplus(x):
    # numerically safe softplus
    x = np.clip(x, -50.0, 50.0)
    return np.log1p(np.exp(x))

def sawtooth_pos(t, v, L):
    # triangle wave path without trig
    # reflect between [0, L)
    s = (v * t) % (2 * L)
    return L - abs(s - L)  # 0..L

def safe_norm(x):
    s = float(np.sum(x))
    return x / s if s > 0 else x

# ---------------- connection ----------------
@dataclass
class Conn:
    # One generic connection — role emerges from drives
    loc: float       # spatial anchor (0..space)
    reach: float     # local kernel width
    alpha: float     # activation scale
    harvest_dir: float = 0.0
    harvest_ind: float = 0.0
    harvest_mot: float = 0.0
    act_cost_accum: float = 0.0
    active_frac_accum: float = 0.0
    age: int = 0

    def kernel(self, x, space):
        # compact bump (no trig): triangular kernel
        d = np.abs(x - self.loc)
        w = np.clip(1.0 - d / max(self.reach, 1e-6), 0.0, 1.0)
        return w

# ---------------- engine ----------------
class Engine:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.space = int(cfg["space"])
        self.frames = int(cfg["frames"])
        self.rng = rng_global(int(cfg["seed"]))

        # state
        self.t = 0
        self.energy = 50.0  # start pool
        self.S = np.zeros(self.space, dtype=np.float32)

        # environment tracks: (pos0, velocity, amplitude)
        self.tracks = []
        self._init_tracks()

        # connections
        self.conns: list[Conn] = []
        self._init_conns(int(cfg["conn.init_count"]))

        # histories
        self.energy_hist = []
        self.env_hist = []   # list of env fields
        self.subs_hist = []  # list of S copies
        self.conn_hist = []  # list of per-conn activation vectors

    # ---------- init helpers ----------
    def _init_tracks(self):
        L = self.space
        n = int(self.cfg["env.num_tracks"])
        bias = float(self.cfg["env.boundary_bias"])
        amp = float(self.cfg["env.energy"])
        self.tracks.clear()
        for _ in range(n):
            # bias start near boundary if bias>0.5
            near_edge = (self.rng.random() < bias)
            x0 = self.rng.uniform(0, L * 0.12) if near_edge else self.rng.uniform(L * 0.2, L * 0.8)
            # reflect to either edge
            if near_edge and self.rng.random() < 0.5:
                x0 = L - x0
            v = self.rng.uniform(0.5, 1.5) * (1 if self.rng.random() < 0.5 else -1)
            a = amp * self.rng.uniform(0.6, 1.2) / max(n, 1)
            self.tracks.append([x0, v, a])

    def _init_conns(self, count: int):
        L = self.space
        self.conns.clear()
        for _ in range(count):
            c = Conn(
                loc=float(self.rng.uniform(0, L)),
                reach=float(self.rng.uniform(L * 0.02, L * 0.12)),
                alpha=float(self.rng.uniform(0.2, 1.0)),
            )
            self.conns.append(c)

    # ---------- stepping ----------
    def step(self, n_steps: int = 1):
        for _ in range(int(n_steps)):
            self._step_once()

    def _env_field(self, t: int):
        L = self.space
        field = np.zeros(L, dtype=np.float32)

        # moving packets
        for (x0, v, a) in self.tracks:
            x = sawtooth_pos(t, v, L)
            # narrow triangular bump centered at x
            width = max(2.0, 0.06 * L)
            xs = np.arange(L)
            d = np.abs(xs - x)
            w = np.clip(1.0 - d / width, 0.0, 1.0)
            field += a * w.astype(np.float32)

        # small background noise
        noise = self.cfg["env.noise"]
        if noise > 0:
            field += noise * self.rng.random(L, dtype=np.float32)

        return field

    def _step_once(self):
        cfg = self.cfg
        L = self.space
        xs = np.arange(L, dtype=np.float32)

        # 1) environment
        E = self._env_field(self.t)
        self.env_hist.append(E.copy())
        if len(self.env_hist) > self.frames:
            self.env_hist.pop(0)

        # 2) connection activations and economy
        gamma_dir = float(cfg["econ.harvest.direct"])
        gamma_ind = float(cfg["econ.harvest.indirect"])
        gamma_mot = float(cfg["econ.harvest.motor"])
        c_act = float(cfg["econ.cost.activation"])
        c_maint = float(cfg["econ.cost.maintenance"])

        total_act_vec = []  # for history heatmap
        harvested = 0.0
        spent = 0.0

        # indirect signal = deviation of S from its mean (model/structure cue)
        S_mean = float(np.mean(self.S))
        S_devi = np.abs(self.S - S_mean)

        for ci, c in enumerate(self.conns):
            k = c.kernel(xs, L)
            # local probes: direct env & indirect substrate variations
            d_direct = float(np.dot(k, E))
            d_ind = float(np.dot(k, S_devi))
            # a small motor probe proportional to how mismatched substrate is here
            mismatch = float(np.dot(k, np.maximum(E - self.S, 0.0)))
            d_motor = mismatch

            # activation (no trig, softplus for positivity)
            a = softplus(c.alpha * (d_direct + 0.5 * d_ind + 0.25 * d_motor))

            # harvest & costs
            P_dir = gamma_dir * d_direct
            P_ind = gamma_ind * d_ind * (a > 0)
            P_mot = gamma_mot * d_motor

            C_act = c_act * (a * a)
            C_main = c_maint

            net = (P_dir + P_ind + P_mot) - (C_act + C_main)

            # accumulate economy
            harvested += (P_dir + P_ind + P_mot)
            spent += (C_act + C_main)

            # subtle local substrate push (motors): move S toward env around kernel
            mot_push = np.clip(E - self.S, 0.0, None) * k
            self.S += float(cfg["subs.couple_conn"]) * mot_push

            # stats
            c.harvest_dir += P_dir
            c.harvest_ind += P_ind
            c.harvest_mot += P_mot
            c.act_cost_accum += (C_act + C_main)
            c.active_frac_accum += float(a > 0)
            c.age += 1

            total_act_vec.append(a)

        # 3) substrate passive dynamics
        self.S = float(cfg["subs.decay"]) * self.S + float(cfg["subs.couple_env"]) * E
        self.S = np.clip(self.S, 0.0, None)

        self.subs_hist.append(self.S.copy())
        if len(self.subs_hist) > self.frames:
            self.subs_hist.pop(0)

        # 4) energy pool
        self.energy += harvested - spent
        self.energy_hist.append(self.energy)
        if len(self.energy_hist) > self.frames:
            self.energy_hist.pop(0)

        # 5) growth (rare, energy‑conditioned)
        if (self.t % int(cfg["conn.grow_every"]) == 0) and (len(self.conns) < int(cfg["conn.max"])):
            # grow where (E - S) is large (unexploited gradient)
            gap = np.maximum(E - self.S, 0.0)
            p = safe_norm(gap + 1e-12)
            loc = float(self.rng.choice(np.arange(L), p=p))
            reach = float(self.rng.uniform(L * 0.03, L * 0.12))
            alpha = float(self.rng.uniform(0.2, 1.2))
            self.conns.append(Conn(loc=loc, reach=reach, alpha=alpha))

        # 6) connection history (activation proxy)
        act_vec = np.array(total_act_vec, dtype=np.float32) if total_act_vec else np.zeros(1, np.float32)
        self.conn_hist.append(act_vec)
        if len(self.conn_hist) > self.frames:
            self.conn_hist.pop(0)

        self.t += 1

    # ---------- state & plots ----------
    def get_state(self) -> dict:
        # count "dominant role" per connection by largest harvest channel
        role_counts = {"SENSE": 0, "INTERNAL": 0, "MOTOR": 0, "UNLABELED": 0}
        for c in self.conns:
            vals = np.array([c.harvest_dir, c.harvest_ind, c.harvest_mot])
            if np.all(vals == 0):
                role_counts["UNLABELED"] += 1
            else:
                idx = int(np.argmax(vals))
                if idx == 0:
                    role_counts["SENSE"] += 1
                elif idx == 1:
                    role_counts["INTERNAL"] += 1
                else:
                    role_counts["MOTOR"] += 1

        return dict(
            t=self.t,
            energy=self.energy,
            energy_hist=np.array(self.energy_hist, dtype=np.float32),
            env_last=self.env_hist[-1] if self.env_hist else np.zeros(self.space),
            env_hist=np.array(self.env_hist[-min(len(self.env_hist), self.frames):], dtype=np.float32),
            subs_last=self.S.copy(),
            subs_hist=np.array(self.subs_hist[-min(len(self.subs_hist), self.frames):], dtype=np.float32),
            conn_hist=self._pad_conn_hist(),
            role_counts=role_counts,
            cfg=self.cfg,
        )

    def _pad_conn_hist(self):
        """Pad connection activation history to a rectangular array for plotting."""
        if not self.conn_hist:
            return np.zeros((1, 1), dtype=np.float32)
        max_len = max(len(v) for v in self.conn_hist)
        H = np.zeros((len(self.conn_hist), max_len), dtype=np.float32)
        for i, v in enumerate(self.conn_hist):
            H[i, :len(v)] = v
        return H

    # Matplotlib helpers for Streamlit
    def plot_env_history(self):
        hist = self.get_state()["env_hist"]  # shape: [T, L]
        fig, ax = plt.subplots(figsize=(8, 2.4))
        if hist.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.imshow(hist.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("frame"); ax.set_ylabel("space index")
        fig.tight_layout()
        return fig

    def plot_subs_history(self):
        hist = self.get_state()["subs_hist"]
        fig, ax = plt.subplots(figsize=(8, 2.4))
        if hist.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.imshow(hist.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_xlabel("frame"); ax.set_ylabel("space index")
        fig.tight_layout()
        return fig

    def plot_conn_history(self):
        H = self.get_state()["conn_hist"]  # [T, Nconn]
        fig, ax = plt.subplots(figsize=(8, 2.4))
        if H.size == 0:
            ax.text(0.5, 0.5, "No connections", ha="center", va="center")
        else:
            ax.imshow(H.T, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_xlabel("frame"); ax.set_ylabel("connection idx (row order)")
        fig.tight_layout()
        return fig


# ---- public factory for Streamlit app ----
def make_engine(cfg: dict):
    return Engine(cfg)


__all__ = ["default_config", "make_engine", "Engine"]