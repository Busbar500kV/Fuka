# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sim_core import run_sim, default_config

st.set_page_config(page_title="Fuka Simulation", layout="wide")

# ---------- small helpers ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def init_cfg():
    base = default_config()
    # Provide hard defaults in case default_config() changes
    base.setdefault("frames", 1600)
    base.setdefault("space", 192)
    base.setdefault("seed", 0)
    base.setdefault("grow_every", 40)
    base.setdefault("grow_attempts", 3)
    base.setdefault("log_every", 40)
    base.setdefault("env_energy", 1.0)
    base.setdefault("env_persistence", 0.95)
    base.setdefault("env_variability", 0.15)
    base.setdefault("save_drive_trace", True)
    return base

# ---------- session state ----------
if "cfg" not in st.session_state:
    st.session_state.cfg = init_cfg()

cfg = st.session_state.cfg  # work on the same dict

# pre-clamp before widgets to avoid widget min/max exceptions
frames_min, frames_max, frames_step = 400, 20000, 200
space_min,  space_max               = 32, 1024

cfg["frames"] = clamp(int(cfg.get("frames", 1600)), frames_min, frames_max)
cfg["space"]  = clamp(int(cfg.get("space", 192)),   space_min,  space_max)
cfg["seed"]   = int(cfg.get("seed", 0))
cfg["grow_every"]    = clamp(int(cfg.get("grow_every", 40)), 10, 2000)
cfg["grow_attempts"] = clamp(int(cfg.get("grow_attempts", 3)),  1, 64)
cfg["log_every"]     = clamp(int(cfg.get("log_every", 40)),    10, 2000)
cfg["env_energy"]       = float(cfg.get("env_energy", 1.0))
cfg["env_persistence"]  = float(cfg.get("env_persistence", 0.95))
cfg["env_variability"]  = float(cfg.get("env_variability", 0.15))
cfg["save_drive_trace"] = bool(cfg.get("save_drive_trace", True))

# ---------- UI ----------
with st.sidebar:
    st.title("Fuka Controls")

    st.subheader("Core")
    cfg["frames"] = st.number_input(
        "Frames",
        min_value=frames_min, max_value=frames_max, value=cfg["frames"], step=frames_step,
        help="Total frames (simulation steps)"
    )
    cfg["space"] = st.number_input(
        "Space (grid size)",
        min_value=space_min, max_value=space_max, value=cfg["space"], step=32,
        help="Spatial discretization (number of points)"
    )
    cfg["seed"] = st.number_input(
        "Random seed", min_value=0, max_value=10_000_000, value=cfg["seed"], step=1
    )

    st.subheader("Growth")
    cfg["grow_every"] = st.number_input(
        "Grow Every (frames)", min_value=10, max_value=2000,
        value=cfg["grow_every"], step=10,
        help="Try to add connections every N frames"
    )
    cfg["grow_attempts"] = st.number_input(
        "Grow Attempts per event", min_value=1, max_value=64,
        value=cfg["grow_attempts"], step=1
    )
    cfg["log_every"] = st.number_input(
        "Log Every (frames)", min_value=10, max_value=2000,
        value=cfg["log_every"], step=10
    )

    st.subheader("Environment")
    cfg["env_energy"] = st.slider(
        "Free energy level at boundary", min_value=0.0, max_value=10.0,
        value=float(cfg["env_energy"]), step=0.1,
        help="Raises available free energy near boundary → tends to favor motors"
    )
    cfg["env_persistence"] = st.slider(
        "Environment persistence", min_value=0.0, max_value=0.999,
        value=float(cfg["env_persistence"]), step=0.01,
        help="How slowly the environment changes (closer to 1.0 = slower)"
    )
    cfg["env_variability"] = st.slider(
        "Environment variability", min_value=0.0, max_value=1.0,
        value=float(cfg["env_variability"]), step=0.01,
        help="How noisy / structured the environment signal is"
    )

    cfg["save_drive_trace"] = st.checkbox(
        "Save drive trace (bigger logs)", value=cfg["save_drive_trace"]
    )

    run_clicked = st.button("Run Simulation", use_container_width=True)

st.title("Fuka: Free-Energy Simulation")

if run_clicked:
    with st.spinner("Running simulation… this can take a bit for large frames/space"):
        try:
            out = run_sim(cfg)
        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.stop()

    st.success("Done!")

    # --- Energy plot ---
    if "energy_series" in out and len(out["energy_series"]) > 0:
        st.subheader("Global Energy Over Time")
        dfE = pd.DataFrame({
            "frame": np.arange(len(out["energy_series"])),
            "energy": np.array(out["energy_series"], dtype=float)
        })
        st.line_chart(dfE.set_index("frame"))

    # --- Connections table ---
    if "conn_table" in out and len(out["conn_table"]) > 0:
        st.subheader("Final Connection Table")
        dfC = pd.DataFrame(out["conn_table"])
        st.dataframe(dfC, use_container_width=True, height=420)

        # CSV download
        csv = dfC.to_csv(index=False)
        st.download_button("Download connections CSV", csv, "connections.csv", "text/csv")

    # --- Drive trace (optional, can be large) ---
    if cfg.get("save_drive_trace") and "drive_trace_head" in out:
        st.subheader("Drive Trace (head)")
        dfD = pd.DataFrame(out["drive_trace_head"])
        st.dataframe(dfD, use_container_width=True, height=360)

        csvD = dfD.to_csv(index=False)
        st.download_button("Download drive-trace head CSV", csvD, "drive_trace_head.csv", "text/csv")

    # --- Event log head ---
    if "event_log_head" in out and len(out["event_log_head"]) > 0:
        st.subheader("Event Log (head)")
        dfL = pd.DataFrame(out["event_log_head"], columns=["when", "event", "payload"])
        st.dataframe(dfL, use_container_width=True, height=300)

        csvL = dfL.to_csv(index=False)
        st.download_button("Download event-log head CSV", csvL, "event_log_head.csv", "text/csv")

else:
    st.info("Set parameters in the sidebar, then click **Run Simulation**.")