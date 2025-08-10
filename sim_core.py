# --- Streamlit live simulation app ---
# Run: streamlit run streamlit_app.py
import threading, time, queue, math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------
def softplus(x):
    # numerically stable softplus
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] + np.log1p(np.exp(-x[pos]))
    out[~pos] = np.log1p(np.exp(x[~pos]))
    return out

# ---------------------------
# Model pieces (local-only physics; no trig, no gains)
# ---------------------------
@dataclass
class Conn:
    # generalized connection; "role" emerges from which drive dominates
    i: int
    L: float
    T: float
    delay_tau: float
    alpha_tau: float
    age: int = 0
    active_frac: float = 0.0
    energy_contrib: float = 0.0
    dom_drive: str = "UNLABELED"  # SENSE/INTERNAL/MOTOR/UNLABELED

    # scratch
    last_a: float = 0.0

    def activity(self, env_slice, subs_slice, motor_drive, sigma=0.1):
        # local-only “event” detector using laplacian-ish contrast
        # env_slice, subs_slice are small local 1D neighborhoods around L
        # motor_drive is a local scalar effort (non-negative)
        d_env = np.abs(env_slice - env_slice.mean())
        d_sub = np.abs(subs_slice - subs_slice.mean())
        # emergent activation from contrast + small noise + motor drive
        a = d_env.mean() + d_sub.mean() + 0.05 * motor_drive + sigma * np.random.randn()
        return max(0.0, float(a))

# ---------------------------
# Simulation core
# ---------------------------
class Sim:
    def __init__(self, N=200, history=1600, seed=0):
        self.rng = np.random.default_rng(seed)
        self.N = N
        self.t = 0
        self.history = history

        # Fields
        self.E_global = 50.0
        self.env = np.zeros(N, dtype=float)   # environment free energy field
        self.subs = np.zeros(N, dtype=float)  # substrate state

        # Data history (for plots)
        self.E_hist = []
        self.env_hist = np.zeros((history, N))
        self.subs_hist = np.zeros((history, N))
        self.act_hist = []   # per-frame activity per connection

        # Connections
        self.conns: List[Conn] = []
        for k in range(9):  # start with a few boundary-touching strands
            L0 = float(self.rng.uniform(0, 1))  # position (0..1)
            self.conns.append(
                Conn(i=k, L=L0, T=0.0, delay_tau=0.0, alpha_tau=0.5 + 0.05*k)
            )

    # --- environment update (calibratable) ---
    def update_environment(self, p):
        # two sources: boundary-source and deep-source
        x = np.linspace(0, 1, self.N)
        # boundary kernel (higher near 0 and 1)
        k_boundary = (np.exp(-((x)/p.boundary_width)) +
                      np.exp(-((1.0-x)/p.boundary_width)))
        # deep kernel (peak at center when deep_center=0.5, width controls how far energy sits)
        k_deep = np.exp(-0.5*((x - p.deep_center)/max(p.deep_width,1e-6))**2)

        # drift sources over time (local effect via slow diffusion below)
        env_new = (p.env_level *
                   (p.boundary_weight * k_boundary +
                    (1.0 - p.boundary_weight) * k_deep))

        # sparse stochastic “events”
        if p.event_rate > 0.0:
            n_events = self.rng.poisson(p.event_rate)
            for _ in range(n_events):
                j = self.rng.integers(0, self.N)
                env_new[j] += p.event_burst * self.rng.uniform(0.8, 1.2)

        self.env = env_new

    # --- local substrate dynamics ---
    def update_substrate(self, p):
        # diffusion (local), decay (local), injection from environment (local)
        lap = np.zeros_like(self.subs)
        lap[1:-1] = self.subs[:-2] - 2*self.subs[1:-1] + self.subs[2:]
        # Neumann boundaries
        lap[0] = self.subs[1] - self.subs[0]
        lap[-1] = self.subs[-2] - self.subs[-1]
        self.subs += p.diffusion * lap - p.decay * self.subs + p.env_to_subs * self.env
        self.subs = np.clip(self.subs, 0.0, p.subs_cap)

    # --- free energy harvester ---
    def step_connections(self, p):
        # simple 1D neighborhoods for each connection (local interaction)
        win = p.conn_window
        act_frame = []
        P_dir_tot = 0.0
        P_ind_tot = 0.0
        P_mot_tot = 0.0
        C_act_tot = 0.0
        C_main_tot = 0.0

        for c in self.conns:
            pos = int(np.clip(round(c.L*(self.N-1)), 0, self.N-1))
            lo = max(0, pos-win)
            hi = min(self.N, pos+win+1)
            env_slice = self.env[lo:hi]
            subs_slice = self.subs[lo:hi]

            # motor probing (random at first; becomes stronger as activity stabilizes)
            motor_effort = max(0.0, p.motor_base + p.motor_adapt * c.last_a)

            a = c.activity(env_slice, subs_slice, motor_effort, sigma=p.act_noise)
            c.last_a = a
            c.age += 1

            # energy harvest components (all local)
            # direct harvest from nearby env gradient
            d_env = np.abs(env_slice - env_slice.mean()).mean()
            P_dir = p.gamma_direct * a * d_env
            # indirect from substrate alignment
            d_sub = np.abs(subs_slice - subs_slice.mean()).mean()
            P_ind = p.gamma_indirect * a * d_sub
            # motor-to-env conversion when motors are “pumping”
            P_mot = p.gamma_motor * motor_effort * a

            # costs: activation + small maintenance
            C_act = p.c_act * a*a
            C_main = p.c_main * (1.0 + 1.0/max(1e-8, (0.2 + abs(c.alpha_tau))))  # small, local-ish

            net = (P_dir + P_ind + P_mot) - (C_act + C_main)
            self.E_global += net

            # bookkeep
            c.energy_contrib += net
            c.active_frac = 0.99*c.active_frac + 0.01*(a>0)
            act_frame.append(a)
            P_dir_tot += P_dir
            P_ind_tot += P_ind
            P_mot_tot += P_mot
            C_act_tot += C_act
            C_main_tot += C_main

            # slow wander in L (emergent exploration)
            if self.rng.random() < p.wander_prob:
                c.L = float(np.clip(c.L + self.rng.normal(scale=p.wander_sigma), 0.0, 1.0))

        self.act_hist.append(np.array(act_frame, dtype=float))

        # emergent labeling (for display only)
        for c in self.conns:
            # rough attribution based on cumulative positive contributions
            # NOTE: display only; doesn't affect physics
            dir_w = p.gamma_direct
            ind_w = p.gamma_indirect
            mot_w = p.gamma_motor
            m = max(dir_w, ind_w, mot_w)
            if m == mot_w:
                c.dom_drive = "MOTOR"
            elif m == ind_w:
                c.dom_drive = "INTERNAL"
            else:
                c.dom_drive = "SENSE"

    # --- growth & prune (very light; local) ---
    def structural_plasticity(self, p):
        # grow with probability proportional to energy trend; prune if aged & unhelpful
        if len(self.conns) < p.conn_cap and self.rng.random() < p.grow_prob:
            L0 = float(self.rng.uniform(0, 1))
            self.conns.append(
                Conn(i=len(self.conns), L=L0, T=0.0,
                     delay_tau=0.0, alpha_tau=self.rng.uniform(0.4, 0.9))
            )
        # prune worst contributor if too many
        if len(self.conns) > 3 and self.rng.random() < p.prune_prob:
            worst = min(self.conns, key=lambda c: c.energy_contrib)
            if worst.energy_contrib < 0:
                self.conns.remove(worst)

    def frame(self, p):
        self.update_environment(p)
        self.update_substrate(p)
        self.step_connections(p)
        self.structural_plasticity(p)

        # record history (ring buffer)
        idx = self.t % self.history
        self.env_hist[idx] = self.env
        self.subs_hist[idx] = self.subs
        self.E_hist.append(self.E_global)
        self.t += 1

# ---------------------------
# Background runner
# ---------------------------
class Runner:
    def __init__(self, sim: Sim, params):
        self.sim = sim
        self.params = params
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self._stop.clear()
            self._pause.clear()
            self.thread.start()

    def stop(self):
        self._stop.set()

    def pause(self, flag: bool):
        if flag:
            self._pause.set()
        else:
            self._pause.clear()

    def _loop(self):
        # run ~20 FPS target (adjustable)
        dt = 0.05
        while not self._stop.is_set():
            if not self._pause.is_set():
                self.sim.frame(self.params)
            time.sleep(dt)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Free‑Energy Simulation", layout="wide")

# sidebar controls (editable live)
with st.sidebar:
    st.header("Controls")

    # Environment calibration
    env_level = st.slider("Environment energy level", 0.0, 20.0, 4.0, 0.1)
    boundary_weight = st.slider("Boundary weight (0=deep,1=boundary)", 0.0, 1.0, 0.6, 0.01)
    boundary_width = st.slider("Boundary width", 0.01, 0.5, 0.08, 0.01)
    deep_center = st.slider("Deep center (0..1)", 0.0, 1.0, 0.5, 0.01)
    deep_width = st.slider("Deep width", 0.02, 0.5, 0.18, 0.01)
    event_rate = st.slider("Event rate (Poisson per frame)", 0.0, 10.0, 1.5, 0.1)
    event_burst = st.slider("Event burst energy", 0.0, 5.0, 1.0, 0.1)

    # Substrate dynamics
    diffusion = st.slider("Diffusion", 0.0, 0.5, 0.08, 0.01)
    decay = st.slider("Decay", 0.0, 0.2, 0.02, 0.01)
    env_to_subs = st.slider("Env -> Substrate coupling", 0.0, 1.0, 0.3, 0.01)
    subs_cap = st.slider("Substrate cap", 0.1, 2.0, 0.8, 0.05)

    # Harvest & costs
    gamma_direct = st.slider("γ_direct", 0.0, 2.0, 0.8, 0.05)
    gamma_indirect = st.slider("γ_indirect", 0.0, 2.0, 0.6, 0.05)
    gamma_motor = st.slider("γ_motor (pumping efficiency)", 0.0, 2.0, 0.4, 0.05)
    c_act = st.slider("Cost: activation", 0.0, 1.5, 0.2, 0.01)
    c_main = st.slider("Cost: maintenance", 0.0, 2.0, 0.3, 0.01)

    # Motors & exploration
    motor_base = st.slider("Motor base effort", 0.0, 1.0, 0.05, 0.01)
    motor_adapt = st.slider("Motor adapt (↑ with activity)", 0.0, 1.0, 0.15, 0.01)
    wander_prob = st.slider("Wander prob per frame", 0.0, 0.2, 0.02, 0.005)
    wander_sigma = st.slider("Wander sigma (pos drift)", 0.0, 0.1, 0.01, 0.001)

    # Plasticity
    conn_cap = st.slider("Max connections", 3, 64, 24, 1)
    grow_prob = st.slider("Grow probability/frame", 0.0, 0.2, 0.02, 0.005)
    prune_prob = st.slider("Prune probability/frame", 0.0, 0.2, 0.01, 0.005)

    # Noise & numerics
    act_noise = st.slider("Activity noise σ", 0.0, 0.5, 0.1, 0.01)
    conn_window = st.slider("Conn local window (cells)", 1, 12, 4, 1)

    st.markdown("---")
    start = st.button("▶️ Start / Resume")
    pause = st.button("⏸️ Pause")
    reset = st.button("🔄 Reset")

# bundle params in a simple namespace object
class P: pass
p = P()
for k, v in dict(
    env_level=env_level, boundary_weight=boundary_weight, boundary_width=boundary_width,
    deep_center=deep_center, deep_width=deep_width, event_rate=event_rate,
    event_burst=event_burst, diffusion=diffusion, decay=decay, env_to_subs=env_to_subs,
    subs_cap=subs_cap, gamma_direct=gamma_direct, gamma_indirect=gamma_indirect,
    gamma_motor=gamma_motor, c_act=c_act, c_main=c_main, motor_base=motor_base,
    motor_adapt=motor_adapt, wander_prob=wander_prob, wander_sigma=wander_sigma,
    conn_cap=conn_cap, grow_prob=grow_prob, prune_prob=prune_prob,
    act_noise=act_noise, conn_window=conn_window
).items():
    setattr(p, k, v)

# session state: sim + runner
if "sim" not in st.session_state:
    st.session_state.sim = Sim(N=200, history=1600, seed=42)
if "runner" not in st.session_state:
    st.session_state.runner = Runner(st.session_state.sim, p)

runner: Runner = st.session_state.runner
runner.params = p  # live-update parameters used in background loop

if start:
    runner.pause(False)
    runner.start()
if pause:
    runner.pause(True)
if reset:
    runner.stop()
    st.session_state.sim = Sim(N=200, history=1600, seed=np.random.randint(0, 10_000))
    st.session_state.runner = Runner(st.session_state.sim, p)
    runner = st.session_state.runner
    runner.start()

# auto-refresh every second while running
st.autorefresh(interval=1000, key="auto")

sim = st.session_state.sim

# ---------------------------
# Plots
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Global Energy Pool")
    fig, ax = plt.subplots(figsize=(6,3))
    if len(sim.E_hist) > 0:
        ax.plot(sim.E_hist)
    ax.set_xlabel("frame")
    ax.set_ylabel("energy")
    st.pyplot(fig)

with col2:
    st.subheader("Environment events (direct field)")
    fig, ax = plt.subplots(figsize=(6,3))
    im = ax.imshow(sim.env_hist.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("frame")
    ax.set_ylabel("space index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Substrate state S")
    fig, ax = plt.subplots(figsize=(6,3))
    im = ax.imshow(sim.subs_hist.T, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("frame")
    ax.set_ylabel("space index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

with col4:
    st.subheader("Connections (rows) — activation proxy")
    fig, ax = plt.subplots(figsize=(6,3))
    if len(sim.act_hist) > 0:
        A = np.stack(sim.act_hist, axis=0)  # [frames, conns]
        im = ax.imshow(A.T, origin="lower", aspect="auto", cmap="coolwarm")
        ax.set_xlabel("frame")
        ax.set_ylabel("connection idx (row order)")
        fig.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No activity yet", ha="center", va="center")
    st.pyplot(fig)

# Emergent role counts
st.subheader("Emergent roles (by dominant drive)")
roles = ["SENSE", "INTERNAL", "MOTOR", "UNLABELED"]
counts = {r:0 for r in roles}
for c in sim.conns:
    counts[c.dom_drive] = counts.get(c.dom_drive, 0) + 1
fig, ax = plt.subplots(figsize=(6,3))
ax.bar(roles, [counts[r] for r in roles])
st.pyplot(fig)

# Final table preview
st.subheader("Connections snapshot")
rows = []
for c in sim.conns[:64]:
    rows.append([c.i, c.dom_drive, round(c.L,3), round(c.T,3), round(c.alpha_tau,3),
                 round(c.active_frac,3), round(c.energy_contrib,3), c.age])
import pandas as pd
df = pd.DataFrame(rows, columns=["idx","role","L","T","alpha","active_frac","energy_sum","age"])
st.dataframe(df, height=240)