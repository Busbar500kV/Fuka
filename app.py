# app.py
# Streamlit UI for Fuka free-energy sim.
# - Sidebar controls for key params
# - JSON editor for env.sources
# - Plots: curves + env heatmap + substrate heatmap
# - Automatic env.frames sync to frames

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sim_core import run_sim, default_config

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

st.title("Fuka — Free‑Energy Gradient Playground")

# ---- Session config
if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()
cfg = st.session_state.cfg  # plain dict

with st.sidebar:
    st.header("Simulation Parameters")

    # Numeric inputs (safe ranges)
    cfg["seed"]   = st.number_input("Seed", min_value=0, max_value=10_000, value=int(cfg.get("seed", 0)), step=1)
    cfg["frames"] = st.number_input("Frames", min_value=200, max_value=50_000, value=int(cfg.get("frames", 5000)), step=200)
    cfg["space"]  = st.number_input("Space (substrate cells)", min_value=16, max_value=2048, value=int(cfg.get("space", 64)), step=16)

    st.markdown("---")
    cfg["k_flux"]  = st.number_input("k_flux (boundary pump)", 0.0, 10.0, value=float(cfg.get("k_flux", 0.05)), step=0.01)
    cfg["k_motor"] = st.number_input("k_motor (motor exploration)", 0.0, 10.0, value=float(cfg.get("k_motor", 2.0)), step=0.05)
    cfg["k_noise"] = st.number_input("k_noise (additional band noise)", 0.0, 2.0, value=float(cfg.get("k_noise", 0.00)), step=0.01)
    cfg["decay"]   = st.number_input("decay", 0.0, 1.0, value=float(cfg.get("decay", 0.01)), step=0.005)
    cfg["diffuse"] = st.number_input("diffuse", 0.0, 1.0, value=float(cfg.get("diffuse", 0.05)), step=0.005)
    cfg["band"]    = st.number_input("boundary band (cells)", 1, 64, value=int(cfg.get("band", 3)), step=1)

    st.markdown("---")
    st.subheader("Environment")
    env = cfg.setdefault("env", {})
    env["length"]      = st.number_input("env.length (cells)", min_value=32, max_value=8192, value=int(env.get("length", 512)), step=32)
    env["noise_sigma"] = st.number_input("env.noise_sigma", 0.0, 1.0, value=float(env.get("noise_sigma", 0.01)), step=0.01)

    st.markdown("**Env Sources (JSON)**")
    default_sources = [
        {"kind": "constant", "amp": 1.0, "start": 8, "width": 10},
        {"kind": "moving_peak", "amp": 0.6, "speed": 0.08, "width": 6.0, "start": 180}
    ]
    sources_text = st.text_area(
        "Edit sources JSON and click Run. (Frames syncs to top-level Frames.)",
        value=json.dumps(env.get("sources", default_sources), indent=2),
        height=220,
    )

    parse_ok = True
    try:
        env["sources"] = json.loads(sources_text)
    except Exception as e:
        parse_ok = False
        st.error(f"Invalid JSON for env.sources: {e}")

    run_btn = st.button("Run Simulation", use_container_width=True, disabled=not parse_ok)

# Sync env.frames to frames so shapes always match
cfg["env"]["frames"] = int(cfg["frames"])

# ---- Run and visualize
if run_btn and parse_ok:
    with st.spinner("Simulating..."):
        hist, env_arr, S = run_sim(cfg)

    st.success("Done!")

    # Curves
    df = pd.DataFrame({
        "t": hist.t,
        "E_cell": hist.E_cell,
        "E_env": hist.E_env,
        "E_flux": hist.E_flux,
    })
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Energies over time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["t"], y=df["E_cell"], mode="lines", name="E_cell"))
        fig.add_trace(go.Scatter(x=df["t"], y=df["E_env"],  mode="lines", name="E_env"))
        fig.add_trace(go.Scatter(x=df["t"], y=df["E_flux"], mode="lines", name="E_flux"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmaps
    with c2:
        st.subheader("Environment (time × env.length)")
        fig_env = go.Figure(data=go.Heatmap(z=env_arr, colorscale="Viridis"))
        fig_env.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_env, use_container_width=True)

    st.subheader("Substrate S (time × space)")
    fig_S = go.Figure(data=go.Heatmap(z=S, colorscale="Plasma"))
    fig_S.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_S, use_container_width=True)

    st.caption(f"Shapes — env: {env_arr.shape}, substrate: {S.shape}.  "
               f"Frames synchronized to {cfg['frames']}.")

else:
    st.info("Set parameters in the sidebar, edit env sources JSON if you like, and click **Run Simulation**.")