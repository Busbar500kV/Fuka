# app.py
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# only import symbols that exist
from sim_core import default_config, make_engine

st.set_page_config(page_title="Fuka — Free‑Energy Gradient Playground", layout="wide")
st.title("Fuka — Free‑Energy Gradient Playground")

# ---------- helpers ----------
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

# ---------- sidebar: SAME knobs as before ----------
with st.sidebar:
    st.header("Parameters")

    cfg = default_config()
    # defaults matching your last good run
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

# Build the plain dict config from UI
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
try:
    user_cfg["env"]["sources"] = json.loads(src_text)
    if not isinstance(user_cfg["env"]["sources"], list):
        raise ValueError("env.sources must be a list")
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

# ---------- run loop ----------
def run_live():
    engine = make_engine(user_cfg)
    T = engine.T
    chunk = int(live_chunk)

    t_hist, cell_hist, env_hist, flux_hist = [], [], [], []

    def redraw(t):
        t_hist.append(t)
        cell_hist.append(engine.hist.E_cell[-1])
        env_hist.append(engine.hist.E_env[-1])
        flux_hist.append(engine.hist.E_flux[-1])

        # resample current env row (same logic as core)
        e_row = engine.env[t]
        if engine.env.shape[1] != engine.X:
            idx = (np.arange(engine.X) * engine.env.shape[1] // engine.X) % engine.env.shape[1]
            e_row = e_row[idx]
        s_row = engine.S[t]

        env_sub_plot.plotly_chart(plot_env_substrate(e_row, s_row), use_container_width=True)
        energy_plot.plotly_chart(plot_energy(t_hist, cell_hist, env_hist, flux_hist), use_container_width=True)
        status.info(f"t = {t+1} / {T}")

    last = [-1]
    def cb(t):
        if t - last[0] >= chunk or t == T - 1:
            last[0] = t
            redraw(t)

    engine.run(progress_cb=cb, snapshot_every=chunk)  # snapshot_every is accepted/ignored
    if last[0] != T - 1:
        redraw(T - 1)

if run_btn:
    run_live()
else:
    st.caption("Adjust knobs, then press **Run / Rerun**.")