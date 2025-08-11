# app.py
# Streamlit UI with live streaming plots for the free-energy simulation.

import json
import time
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim_core import make_engine, default_config, Engine


st.set_page_config(page_title="Fuka: Free‑Energy Live", layout="wide")

st.title("Fuka — Free‑Energy Gradient Simulation (Live)")

# -------------------------
# Session state scaffolding
# -------------------------
if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()
if "engine" not in st.session_state:
    st.session_state.engine = make_engine(st.session_state.cfg)
if "run_live" not in st.session_state:
    st.session_state.run_live = False
if "last_render_t" not in st.session_state:
    st.session_state.last_render_t = -1

cfg: Dict = st.session_state.cfg
engine: Engine = st.session_state.engine

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.subheader("Parameters")

    seed    = st.number_input("seed", 0, 10_000, int(cfg.get("seed", 0)), 1)
    frames  = st.number_input("frames (timeline budget)", 200, 200_000, int(cfg.get("frames", 5000)), 200)
    space   = st.number_input("space (substrate cells)", 8, 1024, int(cfg.get("space", 64)), 1)
    band    = st.number_input("boundary band width", 1, 32, int(cfg.get("band", 3)), 1)

    k_flux  = st.number_input("k_flux (env→cell pump)", 0.0, 10.0, float(cfg.get("k_flux", 0.12)), 0.01)
    k_motor = st.number_input("k_motor (motor pokes)",   0.0, 10.0, float(cfg.get("k_motor", 0.8)), 0.01)
    diffuse = st.number_input("diffuse",                  0.0, 1.0,  float(cfg.get("diffuse", 0.08)), 0.005)
    decay   = st.number_input("decay",                    0.0, 1.0,  float(cfg.get("decay", 0.01)), 0.001)

    st.markdown("---")
    st.caption("Environment (space-time field)")
    env_len   = st.number_input("env.length", 8, 4096, int(cfg["env"].get("length", 512)), 1)
    env_noise = st.number_input("env.noise_sigma", 0.0, 1.0, float(cfg["env"].get("noise_sigma", 0.005)), 0.001)
    env_frames= st.number_input("env.frames (optional, default=frames)", 0, 200_000,
                                int(cfg["env"].get("frames", int(frames))), 200)

    st.caption("Env sources JSON (array)")
    default_sources = cfg["env"].get("sources", [
        {"kind": "moving_peak", "amp": 2.0, "speed": 0.0, "width": 6.0, "start": 15}
    ])
    sources_text = st.text_area(
        "sources",
        value=json.dumps(default_sources, indent=2),
        height=220
    )

    col_btns = st.columns(3)
    with col_btns[0]:
        reset_clicked = st.button("🔁 Reset / Build", use_container_width=True)
    with col_btns[1]:
        step_clicked = st.button("▶️ Step 100", use_container_width=True)
    with col_btns[2]:
        run_clicked = st.toggle("🏃 Run Live", value=st.session_state.run_live)

# Sync toggle back to session_state
st.session_state.run_live = run_clicked

# Update cfg from sidebar widgets
cfg["seed"]   = int(seed)
cfg["frames"] = int(frames)
cfg["space"]  = int(space)
cfg["band"]   = int(band)
cfg["k_flux"] = float(k_flux)
cfg["k_motor"]= float(k_motor)
cfg["diffuse"]= float(diffuse)
cfg["decay"]  = float(decay)

# Parse sources JSON (safe default on error)
try:
    parsed_sources = json.loads(sources_text)
    if not isinstance(parsed_sources, list):
        raise ValueError("sources must be a JSON list")
except Exception as e:
    st.warning(f"Invalid sources JSON, using default. Error: {e}")
    parsed_sources = [{"kind": "moving_peak", "amp": 2.0, "speed": 0.0, "width": 6.0, "start": 15}]

cfg["env"]["length"] = int(env_len)
cfg["env"]["noise_sigma"] = float(env_noise)
cfg["env"]["frames"] = int(env_frames) if env_frames > 0 else int(frames)
cfg["env"]["sources"] = parsed_sources

# Rebuild engine if requested
if reset_clicked:
    st.session_state.engine = make_engine(cfg)
    st.session_state.last_render_t = -1
    engine = st.session_state.engine
    st.success("Engine rebuilt ✅")

# -------------------------
# Live plotting placeholders
# -------------------------
ph_top = st.container()
with ph_top:
    colA, colB = st.columns([1, 1])
    with colA:
        env_ph = st.empty()
    with colB:
        sub_ph = st.empty()

ph_bottom = st.container()
with ph_bottom:
    curves_ph = st.empty()

# -------------------------
# Render helpers
# -------------------------

def plot_env(env_matrix: np.ndarray, t_max: int):
    # Slice up to t_max (wrap-safe)
    t_max = max(0, int(t_max))
    rows = min(t_max + 1, env_matrix.shape[0])
    data = env_matrix[:rows, :]
    fig = go.Figure(data=go.Heatmap(z=data, colorscale="Viridis", colorbar=dict(title="env")))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="x", yaxis_title="t")
    return fig

def plot_substrate(S: np.ndarray, t_max: int):
    rows = min(t_max + 1, S.shape[0])
    data = S[:rows, :]
    fig = go.Figure(data=go.Heatmap(z=data, colorscale="Plasma", colorbar=dict(title="substrate")))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="x", yaxis_title="t")
    return fig

def plot_curves(hist_t, E_cell, E_env, E_flux):
    df = pd.DataFrame({
        "t": hist_t,
        "E_cell": E_cell,
        "E_env": E_env,
        "E_flux": E_flux
    })
    fig = go.Figure()
    fig.add_scatter(x=df["t"], y=df["E_cell"], name="E_cell")
    fig.add_scatter(x=df["t"], y=df["E_env"],  name="E_env")
    fig.add_scatter(x=df["t"], y=df["E_flux"], name="E_flux", line=dict(dash="dot"))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="t")
    return fig

def redraw(engine: Engine):
    t_now = engine.t - 1
    if t_now < 0:
        t_now = 0
    env_fig = plot_env(engine.env, t_now)
    sub_fig = plot_substrate(engine.S, t_now)
    env_ph.plotly_chart(env_fig, use_container_width=True)
    sub_ph.plotly_chart(sub_fig, use_container_width=True)
    curves_ph.plotly_chart(
        plot_curves(engine.hist.t, engine.hist.E_cell, engine.hist.E_env, engine.hist.E_flux),
        use_container_width=True
    )
    st.session_state.last_render_t = t_now

# Initial draw (if nothing drawn yet)
if st.session_state.last_render_t < 0:
    redraw(engine)

# -------------------------
# Live / Step controls
# -------------------------

# Step 100 frames on demand
if step_clicked:
    engine.run_chunk(100)
    redraw(engine)

# Continuous live run (stream)
if st.session_state.run_live:
    # Run in small batches so UI stays responsive
    batch = 20
    # Keep loop short; the script has to finish for Streamlit to process events.
    # We simulate "continuous" by doing a few batches per run and letting the
    # script rerun when the toggle is still on.
    for _ in range(10):
        engine.run_chunk(batch)
        redraw(engine)
        time.sleep(0.05)

    # If you really want a harder continuous mode, increase range() above,
    # but keep in mind Streamlit must yield control back periodically.