# app.py
# Streamlit UI that drives sim_core.Engine with safe streaming updates.

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

# --- Import sim_core safely and show diagnostics so we know what got loaded ---
import sim_core as sc  # module
from sim_core import make_engine, default_config, Engine  # symbols

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

st.title("Fuka – Free‑Energy Gradient Simulation (Live)")
st.caption("Continuous run with on-the-fly plotting. No nonlocal, no syntax traps.")

# ----- Sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    cfg: Dict = default_config()

    # Guardrails for min/max to avoid Streamlit's 'below min' errors
    frames_min, frames_max, frames_step = 200, 20000, 100
    space_min, space_max = 32, 1024

    cfg["frames"] = int(
        st.number_input("Frames", min_value=frames_min, max_value=frames_max,
                        value=int(cfg["frames"]), step=frames_step)
    )
    cfg["space"] = int(
        st.number_input("Space (DOF)", min_value=space_min, max_value=space_max,
                        value=int(cfg["space"]), step=32)
    )
    cfg["seed"] = int(st.number_input("Seed", min_value=0, max_value=1_000_000,
                                      value=int(cfg["seed"]), step=1))

    st.markdown("**Free‑energy & tendencies**")
    cfg["env_level"] = float(st.slider("Environment level", 0.05, 5.0, float(cfg["env_level"]), 0.05))
    cfg["sense_bias"] = float(st.slider("Sense bias", 0.0, 1.0, float(cfg["sense_bias"]), 0.01))
    cfg["motor_bias"] = float(st.slider("Motor bias", 0.0, 1.0, float(cfg["motor_bias"]), 0.01))
    cfg["internal_bias"] = float(st.slider("Internal bias", 0.0, 1.0, float(cfg["internal_bias"]), 0.01))

    st.markdown("**Live mode**")
    live_mode = st.checkbox("Live streaming (update during run)", value=True)
    cfg["chunk"] = int(st.number_input("Redraw every N ticks", min_value=1, max_value=500,
                                       value=int(cfg["chunk"]), step=1))
    cfg["sleep_ms"] = int(st.number_input("Artificial delay per tick (ms)", min_value=0, max_value=50,
                                          value=int(cfg.get("sleep_ms", 0)), step=1))

    run_btn = st.button("Run simulation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Import diagnostics**")
    st.code(
        f"sim_core path: {Path(sc.__file__).as_posix()}\n"
        f"has make_engine: {hasattr(sc, 'make_engine')}\n"
        f"has Engine:      {hasattr(sc, 'Engine')}\n"
        f"default_config(): {default_config().__class__.__name__}",
        language="text",
    )

# ----- Placeholders for outputs ----------------------------------------------
col_top, col_bot = st.columns([2, 1], gap="large")
with col_top:
    status = st.empty()
    chart = st.empty()
with col_bot:
    table_box = st.empty()

log_box = st.expander("Event log", expanded=False)
log_area = log_box.empty()

# ----- Helpers ----------------------------------------------------------------
def redraw(engine: Engine, t: int):
    """Update chart + table safely up to tick t."""
    df = engine.curves_frame(upto=t)
    if not df.empty:
        # Energy chart
        chart.line_chart(
            df.set_index("tick")[["E_env", "E_cell", "E_flux"]],
            height=280
        )
    # Show the latest 'pressures' in a tiny table
    latest = df.iloc[[-1]][["Sense", "Motor", "Internal", "E_cell", "E_env", "E_flux"]].copy() if not df.empty else pd.DataFrame()
    table_box.dataframe(latest, use_container_width=True)

def append_log(msg: str):
    prev = st.session_state.get("log_txt", "")
    new = (prev + msg + "\n")[-8000:]
    st.session_state["log_txt"] = new
    log_area.code(new, language="text")

# ----- Run button action ------------------------------------------------------
if run_btn:
    # Create engine fresh each time you click Run
    engine = make_engine(cfg)

    # Tell user what’s happening
    status.info("Starting simulation …")

    try:
        if live_mode:
            # mutable holder (no `nonlocal`)
            last = [0]
            chunk = max(1, int(cfg["chunk"]))

            def cb(t: int):
                # stream updates in chunks (or on the final tick)
                if t - last[0] >= chunk or t == cfg["frames"] - 1:
                    last[0] = t
                    append_log(f"tick {t}")
                    redraw(engine, t)

            engine.run(progress_cb=cb)
            redraw(engine, cfg["frames"] - 1)
        else:
            engine.run(progress_cb=None)
            redraw(engine, cfg["frames"] - 1)

        status.success("Done!")
    except Exception as e:
        status.error("Simulation failed.")
        st.exception(e)

st.caption("Tip: turn on **Live streaming** and optionally set a small per‑tick delay (1–5 ms) to watch it evolve.")