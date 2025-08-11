# app.py
# Streamlit UI that builds Engine from sim_core and streams live updates.

import json
import time
from typing import Dict

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Import only symbols that actually exist
from sim_core import Engine, default_config, Config, FieldCfg

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")
st.title("Fuka — Free‑Energy Gradient Playground")

# ---------- helpers ----------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def build_engine_from_dict(cfg: Dict) -> Engine:
    # mirror sim_core.run_sim config build
    env_cfg = cfg.get("env", {})
    fcfg = FieldCfg(
        length=int(env_cfg.get("length", 512)),
        frames=int(env_cfg.get("frames", cfg.get("frames", 5000))),
        noise_sigma=float(env_cfg.get("noise_sigma", 0.01)),
        sources=env_cfg.get("sources", FieldCfg().sources),
    )
    ecfg = Config(
        seed=int(cfg.get("seed", 0)),
        frames=int(cfg.get("frames", 5000)),
        space=int(cfg.get("space", 64)),
        n_init=int(cfg.get("n_init", 9)),
        k_flux=float(cfg.get("k_flux", 0.05)),
        k_motor=float(cfg.get("k_motor", 2.0)),
        decay=float(cfg.get("decay", 0.01)),
        diffuse=float(cfg.get("diffuse", 0.05)),
        env=fcfg,
    )
    return Engine(ecfg)

def plot_env_substrate(env_row: np.ndarray, subs_row: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=env_row, name="Env (row t)", mode="lines"))
    fig.add_trace(go.Scatter(y=subs_row, name="Substrate S (row t)", mode="lines"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def plot_energy(t_hist, cell_hist, env_hist, flux_hist):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_hist, y=cell_hist, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t_hist, y=env_hist,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t_hist, y=flux_hist, name="E_flux", mode="lines"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------- sidebar (knobs) ----------

with st.sidebar:
    st.header("Parameters")

    cfg = default_config()
    # Defaults close to your experiment
    cfg["space"]  = 64
    cfg["frames"] = 2000
    cfg["k_flux"] = 0.08
    cfg["k_motor"]= 1.5
    cfg["diffuse"]= 0.08
    cfg["decay"]  = 0.01
    cfg["env"]["length"] = 512
    cfg["env"]["frames"] = cfg["frames"]
    cfg["env"]["noise_sigma"] = 0.005
    cfg["env"]["sources"] = [
        {"kind": "moving_peak", "amp": 1.2, "speed": 0.02, "width": 6.0, "start": 20}
    ]

    # UI controls
    seed     = st.number_input("seed",   0, 10_000, value=int(cfg["seed"]), step=1)
    frames   = st.number_input("frames", 200, 50_000, value=int(cfg["frames"]), step=200)
    space    = st.number_input("space",   8, 1024,   value=int(cfg["space"]),  step=8)

    k_flux   = st.slider("k_flux (env→cell pump)", 0.0, 1.0, float(cfg["k_flux"]), 0.01)
    k_motor  = st.slider("k_motor (motor randomness)", 0.0, 5.0, float(cfg["k_motor"]), 0.05)
    diffuse  = st.slider("diffuse (substrate)", 0.0, 1.0, float(cfg["diffuse"]), 0.01)
    decay    = st.slider("decay (substrate)",   0.0, 0.2, float(cfg["decay"]),   0.005)

    st.markdown("**Environment (space–time)**")
    env_len  = st.number_input("env.length", 8, 4096, value=int(cfg["env"]["length"]), step=8)
    env_noise= st.slider("env.noise_sigma", 0.0, 0.2, float(cfg["env"]["noise_sigma"]), 0.001)

    st.caption("Sources JSON (list)")
    default_src = json.dumps(cfg["env"]["sources"], indent=2)
    src_text = st.text_area("env.sources", value=default_src, height=160)

    live_chunk = st.number_input("Live update chunk (frames)", 10, 2000, value=150, step=10)

    run_btn = st.button("Run / Rerun", type="primary")

# Build config dict from UI
user_cfg = {
    "seed": int(seed),
    "frames": int(frames),
    "space": int(space),
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "diffuse": float(diffuse),
    "decay": float(decay),
    "env": {
        "length": int(env_len),
        "frames": int(frames),
        "noise_sigma": float(env_noise),
    },
}
# parse sources json safely
try:
    user_cfg["env"]["sources"] = json.loads(src_text)
    if not isinstance(user_cfg["env"]["sources"], list):
        raise ValueError("sources must be a list of dicts")
except Exception as e:
    st.warning(f"Invalid sources JSON. Using default. ({e})")
    user_cfg["env"]["sources"] = [
        {"kind": "moving_peak", "amp": 1.2, "speed": 0.02, "width": 6.0, "start": 20}
    ]

# ---------- layout ----------

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Env vs Substrate (live)")
    env_sub_plot = st.empty()
with col2:
    st.subheader("Energy (live)")
    energy_plot = st.empty()

status = st.empty()

# ---------- run / live stream ----------

def run_live():
    engine = build_engine_from_dict(user_cfg)
    T = engine.T
    chunk = int(live_chunk)

    t_hist, cell_hist, env_hist, flux_hist = [], [], [], []

    def redraw(t):
        # time histories
        t_hist.append(t)
        cell_hist.append(engine.hist.E_cell[-1])
        env_hist.append(engine.hist.E_env[-1])
        flux_hist.append(engine.hist.E_flux[-1])

        # plots
        # current env row resampled (same logic as in engine.step)
        e_row = engine.env[t]
        if engine.env.shape[1] != engine.X:
            idx = (np.arange(engine.X) * engine.env.shape[1] // engine.X) % engine.env.shape[1]
            e_row = e_row[idx]
        s_row = engine.S[t]

        env_sub_plot.plotly_chart(plot_env_substrate(e_row, s_row), use_container_width=True)
        energy_plot.plotly_chart(plot_energy(t_hist, cell_hist, env_hist, flux_hist), use_container_width=True)
        status.info(f"t = {t+1} / {T}")

    last = [-1]  # tiny “box” to allow inner function to mutate

    def cb(t):
        if t - last[0] >= chunk or t == T - 1:
            last[0] = t
            redraw(t)

    # IMPORTANT: Engine accepts snapshot_every but ignores it, to stay compatible.
    engine.run(progress_cb=cb, snapshot_every=chunk)

    # final draw if needed
    if last[0] != T - 1:
        redraw(T - 1)

if run_btn:
    run_live()
else:
    st.caption("Adjust knobs on the left, then press **Run / Rerun**.")

st.markdown("---")
st.caption("Tip: If you ever see an import error again, Streamlit might be holding an old bytecode in memory. Click the Rerun button on the top-right or reboot the app.")