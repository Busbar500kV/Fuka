# app.py
import time
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from sim_core import default_config, make_engine
# top of app.py
from sim_core import default_config, Engine

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

# ---------------- Session state ----------------
if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()
if "engine" not in st.session_state:
    # where we create the engine
    st.session_state.engine = Engine(cfg)
    # st.session_state.engine = None   # created on Play/Reset
if "running" not in st.session_state:
    st.session_state.running = False

cfg = st.session_state.cfg

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.title("Controls")

    # Core
    cfg["space"] = st.number_input("Space (grid size)", 32, 1024, int(cfg.get("space", 192)), 32)
    cfg["seed"]  = st.number_input("Random seed", 0, 10_000_000, int(cfg.get("seed", 0)), 1)

    st.subheader("Environment")
    cfg["env_energy"] = st.slider("Boundary free energy", 0.0, 10.0, float(cfg.get("env_energy", 1.0)), 0.1)
    cfg["env_persistence"] = st.slider("Environment persistence", 0.0, 0.999, float(cfg.get("env_persistence", 0.95)), 0.01)
    cfg["env_variability"] = st.slider("Environment variability", 0.0, 1.0, float(cfg.get("env_variability", 0.15)), 0.01)

    st.subheader("Growth")
    cfg["grow_every"]    = st.number_input("Grow Every (frames)", 10, 2000, int(cfg.get("grow_every", 40)), 10)
    cfg["grow_attempts"] = st.number_input("Grow Attempts per event", 1, 64, int(cfg.get("grow_attempts", 3)), 1)

    st.subheader("Loop")
    cfg["step_chunk"] = st.number_input("Frames per tick", 1, 200, int(cfg.get("step_chunk", 5)), 1)
    tick_ms = st.number_input("Update every (ms)", 50, 2000, 200, 10)

    # Buttons
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("▶ Play", use_container_width=True):
            if st.session_state.engine is None:
                st.session_state.engine = make_engine(cfg)
            # update engine cfg on the fly
            st.session_state.engine.cfg.update(cfg)
            st.session_state.running = True
    with colB:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.running = False
    with colC:
        if st.button("⟲ Reset", use_container_width=True):
            st.session_state.engine = make_engine(cfg)
            st.session_state.running = False

    if st.button("Step once", use_container_width=True):
        if st.session_state.engine is None:
            st.session_state.engine = make_engine(cfg)
        st.session_state.engine.cfg.update(cfg)
        st.session_state.engine.step(int(cfg["step_chunk"]))
        st.experimental_rerun()

st.title("Fuka: Free‑Energy Simulation")

# ---------------- Placeholders ----------------
ph_energy = st.empty()
col1, col2 = st.columns([1.2, 1.0], gap="large")
with col1:
    ph_env = st.empty()
with col2:
    ph_table = st.empty()
ph_log = st.expander("Event Log (head)", expanded=False)
ph_drive = st.expander("Drive Trace (head)", expanded=False)

def render(engine) -> None:
    snap = engine.snapshot()

    # Energy chart
    if snap["energy_series"]:
        dfE = pd.DataFrame({
            "frame": np.arange(len(snap["energy_series"])),
            "energy": np.array(snap["energy_series"], dtype=float)
        })
        ph_energy.subheader("Global Energy (recent)"); ph_energy.line_chart(dfE.set_index("frame"))

    # Environment / substrate heat (env only here, substrate shown via last row intensity)
    with col1:
        if snap["env"] is not None and snap["env"].size:
            ph_env.subheader("Environment (recent window)")
            ph_env.pyplot(_imshow(snap["env"]))

    # Connections table
    dfC = pd.DataFrame(snap["conn_table"])
    with col2:
        ph_table.subheader("Connections (summary)")
        ph_table.dataframe(dfC, use_container_width=True, height=400)

    with ph_log:
        if snap["event_log_head"]:
            dfL = pd.DataFrame(snap["event_log_head"], columns=["when", "event", "payload"])
            st.dataframe(dfL, use_container_width=True, height=220)

    with ph_drive:
        if snap["drive_trace_head"]:
            dfD = pd.DataFrame(snap["drive_trace_head"])
            st.dataframe(dfD, use_container_width=True, height=220)

def _imshow(arr2d):
    # Tiny helper to plot a heatmap with matplotlib on demand
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(arr2d, aspect="auto", origin="lower")
    ax.set_xlabel("recent frames")
    ax.set_ylabel("space")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

# ---------------- Run loop ----------------
engine = st.session_state.engine
if engine is not None:
    render(engine)

# When playing, advance in small chunks and redraw.
# NOTE: Streamlit executes top->bottom each run, so we trigger another rerun.
if st.session_state.running:
    engine.cfg.update(cfg)               # live-apply changed sliders
    engine.step(int(cfg["step_chunk"]))  # advance
    render(engine)
    # gentle pacing
    time.sleep(int(tick_ms)/1000.0)
    st.experimental_rerun()
else:
    st.info("Press **Play** to start continuous simulation. Use sliders while it runs.")