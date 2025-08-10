# app.py
# Streamlit front‑end for the Fuka free‑energy simulation

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import streamlit as st

from sim_core import default_config, make_engine

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

# ---------- helpers ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def get_cfg_from_sidebar():
    cfg = dict(default_config)

    with st.sidebar:
        st.header("Simulation controls")

        # Core sizes and runtime
        frames_min, frames_max, frames_step = 200, 20000, 200
        space_min, space_max = 64, 512

        cfg["frames"] = st.number_input(
            "Max frames to keep (history buffer)",
            min_value=frames_min, max_value=frames_max,
            value=cfg["frames"], step=frames_step
        )
        cfg["space"] = st.number_input(
            "Space size (spatial cells)",
            min_value=space_min, max_value=space_max,
            value=cfg["space"], step=16
        )
        cfg["seed"] = int(st.number_input("Seed", 0, 10_000, cfg["seed"], 1))

        st.markdown("---")
        st.subheader("Environment energy")
        cfg["env.energy"] = st.slider("Free energy (total scale)", 0.0, 10.0, cfg["env.energy"], 0.1)
        cfg["env.boundary_bias"] = st.slider("Boundary bias (0=deep, 1=boundary)", 0.0, 1.0, cfg["env.boundary_bias"], 0.01)
        cfg["env.num_tracks"] = int(st.number_input("Moving energy tracks", 1, 12, cfg["env.num_tracks"], 1))
        cfg["env.noise"] = st.slider("Background noise", 0.0, 0.5, cfg["env.noise"], 0.01)

        st.markdown("---")
        st.subheader("Substrate")
        cfg["subs.decay"] = st.slider("Substrate decay", 0.80, 0.999, cfg["subs.decay"], 0.001)
        cfg["subs.couple_env"] = st.slider("Env→substrate coupling", 0.0, 1.0, cfg["subs.couple_env"], 0.01)
        cfg["subs.couple_conn"] = st.slider("Conn→substrate coupling", 0.0, 1.0, cfg["subs.couple_conn"], 0.01)

        st.markdown("---")
        st.subheader("Connections")
        cfg["conn.init_count"] = int(st.number_input("Initial connections", 0, 64, cfg["conn.init_count"], 1))
        cfg["conn.grow_every"] = int(st.number_input("Grow attempt every N frames", 1, 500, cfg["conn.grow_every"], 1))
        cfg["conn.max"] = int(st.number_input("Max connections", 0, 256, cfg["conn.max"], 1))

        st.markdown("---")
        st.subheader("Economy")
        cfg["econ.cost.activation"] = st.slider("Activation cost", 0.0, 0.1, cfg["econ.cost.activation"], 0.001)
        cfg["econ.cost.maintenance"] = st.slider("Maintenance cost", 0.0, 0.05, cfg["econ.cost.maintenance"], 0.001)
        cfg["econ.harvest.direct"] = st.slider("Direct harvest factor", 0.0, 1.0, cfg["econ.harvest.direct"], 0.01)
        cfg["econ.harvest.indirect"] = st.slider("Indirect (model) harvest", 0.0, 1.0, cfg["econ.harvest.indirect"], 0.01)
        cfg["econ.harvest.motor"] = st.slider("Motor harvest factor", 0.0, 1.0, cfg["econ.harvest.motor"], 0.01)

        st.markdown("---")
        st.subheader("Run settings")
        cfg["ui.steps_per_tick"] = int(st.number_input("Steps per tick", 1, 200, cfg["ui.steps_per_tick"], 1))
        cfg["ui.refresh_ms"] = int(st.number_input("Refresh (ms)", 100, 3000, cfg["ui.refresh_ms"], 50))

        st.caption("Tip: Stop → tweak sliders → Start to apply changes.")

    return cfg

# ---------- session engine ----------
if "engine" not in st.session_state:
    st.session_state.engine = None
if "running" not in st.session_state:
    st.session_state.running = False

# Build or rebuild engine
cfg = get_cfg_from_sidebar()

col_l, col_r = st.columns([1, 1])
with col_l:
    if st.button("Initialize / Reset", use_container_width=True):
        st.session_state.engine = make_engine(cfg)
        st.session_state.running = False
with col_r:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start", use_container_width=True, type="primary"):
            if st.session_state.engine is None:
                st.session_state.engine = make_engine(cfg)
            st.session_state.running = True
    with c2:
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False
    with c3:
        if st.button("Step once", use_container_width=True):
            if st.session_state.engine is None:
                st.session_state.engine = make_engine(cfg)
            st.session_state.engine.step(cfg["ui.steps_per_tick"])

if st.session_state.running:
    # auto-refresh loop while running
    st_autorefresh = st.experimental_rerun  # compatibility alias
    st.session_state.engine.step(cfg["ui.steps_per_tick"])
    # schedule next UI tick
    st.experimental_singleton.clear()  # no-op guard for older runtimes
    st_autorefresh()

# ---------- plots ----------
st.title("Fuka: Free‑Energy Simulation")

eng = st.session_state.engine
if eng is None:
    st.info("Click **Initialize / Reset** to create the engine, then **Start**.")
    st.stop()

state = eng.get_state()

# 1) Energy curve
e_df = pd.DataFrame({"energy": state["energy_hist"]})
st.subheader("Global Energy")
st.line_chart(e_df, height=180)

# 2) Environment field (history)
st.subheader("Environment events (direct field)")
st.caption("Heatmap over time (x‑axis) and space (y‑axis).")
st.pyplot(eng.plot_env_history(), clear_figure=True, use_container_width=True)

# 3) Substrate history
st.subheader("Substrate state S")
st.pyplot(eng.plot_subs_history(), clear_figure=True, use_container_width=True)

# 4) Connections activation history
st.subheader("Connections (rows) — activation proxy")
st.pyplot(eng.plot_conn_history(), clear_figure=True, use_container_width=True)

# 5) Emergent roles
st.subheader("Emergent roles (by dominant harvest)")
st.bar_chart(pd.Series(state["role_counts"]), height=240)

st.success("Done!" if not st.session_state.running else "Running…")