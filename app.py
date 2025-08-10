# app.py
import time
import io
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sim_core import Engine, default_config, FieldCfg

st.set_page_config(page_title="Fuka – Free‑Energy Gradient Simulation (Live)", layout="wide")

# ---------------- state helpers ----------------
def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = default_config()
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "paused" not in st.session_state:
        st.session_state.paused = False
    if "history" not in st.session_state:
        st.session_state.history = None
    if "runs" not in st.session_state:
        st.session_state.runs = {}  # name -> snapshot

init_state()

def new_engine():
    st.session_state.engine = Engine(st.session_state.cfg)
    st.session_state.history = st.session_state.engine.get_series()

# ---------------- sidebar (controls) ----------------
with st.sidebar:
    st.header("Run Controls")

    cfg: Dict[str, Any] = st.session_state.cfg
    # high-level
    cfg["frames"] = st.number_input("Frames", min_value=400, max_value=50000, value=int(cfg["frames"]), step=400)
    cfg["seed"]   = st.number_input("Seed", min_value=0, max_value=10_000, value=int(cfg["seed"]), step=1)
    cfg["redraw_every"] = st.slider("Redraw every N ticks", 1, 200, int(cfg.get("redraw_every", 20)))

    st.subheader("Energy biases")
    cfg["sense_bias"]    = st.slider("Sense bias",    0.0, 1.0, float(cfg.get("sense_bias", 0.3)))
    cfg["motor_bias"]    = st.slider("Motor bias",    0.0, 1.0, float(cfg.get("motor_bias", 0.3)))
    cfg["internal_bias"] = st.slider("Internal bias", 0.0, 1.0, float(cfg.get("internal_bias", 0.4)))

    st.subheader("Environment (space‑time)")
    f = cfg.get("field", {})
    if isinstance(f, dict):
        f = FieldCfg(**f)

    f.space       = st.slider("Space size", 64, 512, int(f.space), step=8)
    f.K           = st.slider("DoF (K kernels)", 1, 24, int(f.K))
    f.amp         = st.slider("Amplitude", 0.1, 4.0, float(f.amp))
    f.width_min   = st.slider("Width min", 2.0, 32.0, float(f.width_min))
    f.width_max   = st.slider("Width max", 4.0, 64.0, float(f.width_max))
    f.speed_min   = st.slider("Speed min", 0.0, 3.0, float(f.speed_min))
    f.speed_max   = st.slider("Speed max", 0.1, 5.0, float(f.speed_max))
    f.boundary_idx = st.slider("Boundary index", 0, f.space-1, int(f.boundary_idx))
    f.offset      = st.slider("Global offset", 0, f.space-1, int(f.offset))
    f.preset      = st.selectbox("Shape preset", ["gauss", "ridge", "blob"], index=["gauss","ridge","blob"].index(f.preset))

    cfg["field"] = f.__dict__  # write back

    st.divider()
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Start / Restart", use_container_width=True):
            new_engine()
            st.session_state.running = True
            st.session_state.paused = False
            st.toast("Started")
    with colB:
        if st.button("Pause/Resume", use_container_width=True):
            if st.session_state.running:
                st.session_state.paused = not st.session_state.paused
                st.toast("Paused" if st.session_state.paused else "Resumed")
    with colC:
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False
            st.session_state.paused = False
            st.toast("Stopped")

    st.subheader("Live Perturbations")
    if st.button("Increase free energy ×1.2"):
        if st.session_state.engine:
            st.session_state.engine.perturb_env(1.2)
            st.toast("Environment boosted")

    st.subheader("Autosave / History")
    run_name = st.text_input("Save name", value=f"run_{int(time.time())}")
    if st.button("Save snapshot"):
        if st.session_state.engine:
            snap = st.session_state.engine.snapshot()
            st.session_state.runs[run_name] = snap
            st.toast(f"Saved {run_name}")
    pick = st.selectbox("Load snapshot", ["—"] + list(st.session_state.runs.keys()))
    if pick != "—":
        snap = st.session_state.runs[pick]
        st.session_state.cfg = snap["cfg"]
        new_engine()
        # load history (we keep current engine, history is just for plots)
        st.session_state.history = {k: np.array(v) for k, v in snap["history"].items()}
        st.toast(f"Loaded {pick}")

st.title("Fuka – Free‑Energy Gradient Simulation (Live)")
status = st.empty()

# --------- charts ---------
charts = st.container()
with charts:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_energy = go.Figure()
        fig_energy.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
        energy_plot = st.plotly_chart(fig_energy, use_container_width=True)
    with c2:
        fig_roles = go.Figure()
        fig_roles.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
        roles_plot = st.plotly_chart(fig_roles, use_container_width=True)

    c3, c4 = st.columns([2, 1])
    with c3:
        env_plot = st.plotly_chart(go.Figure().update_layout(height=240, margin=dict(l=20,r=20,t=10,b=10)),
                                   use_container_width=True)
    with c4:
        # download CSV
        if st.session_state.history is not None:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="fuka_run.csv", mime="text/csv", use_container_width=True)

# --------- drawing helpers ---------
def redraw(tick: int):
    status.success("Running…" if st.session_state.running else "Done!")
    hist = st.session_state.engine.get_series() if st.session_state.engine else st.session_state.history
    if hist is None or len(hist["E_cell"]) == 0:
        return
    st.session_state.history = hist  # keep last for download

    # Energy curves
    x = np.arange(len(hist["E_cell"]))
    fig_energy = go.Figure()
    fig_energy.add_scatter(x=x, y=hist["E_cell"], name="E_cell")
    fig_energy.add_scatter(x=x, y=hist["E_env"],  name="E_env")
    fig_energy.add_scatter(x=x, y=hist["E_flux"], name="E_flux")
    fig_energy.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h"))
    energy_plot.plotly_chart(fig_energy, use_container_width=True)

    # Roles over time
    fig_roles = go.Figure()
    for k in ["SENSE","MOTOR","INTERNAL"]:
        fig_roles.add_scatter(x=x, y=hist[k], name=k)
    fig_roles.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h"))
    roles_plot.plotly_chart(fig_roles, use_container_width=True)

    # Environment snapshot at current tick
    env = st.session_state.engine.get_env_frame(min(tick, len(x)-1)) if st.session_state.engine else None
    if env is not None:
        fig_env = go.Figure()
        fig_env.add_bar(x=list(range(env.size)), y=env, name="env slice")
        fig_env.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=10), showlegend=False)
        env_plot.plotly_chart(fig_env, use_container_width=True)

# --------- main loop (cooperative, Streamlit-friendly) ---------
def run_loop():
    eng = st.session_state.engine
    if not eng:
        return
    chunk = int(st.session_state.cfg.get("redraw_every", 20))
    # run small chunks, then rerun page
    for _ in range(chunk):
        # if paused or stopped, break out
        if not st.session_state.running or st.session_state.paused:
            break
        t_now = len(eng.history["E_cell"])
        if t_now >= eng.frames:
            st.session_state.running = False
            break
        eng.step(t_now)
    redraw(len(eng.history["E_cell"]) - 1)
    # schedule another pass while running
    if st.session_state.running and not st.session_state.paused:
        time.sleep(0.05)
        st.experimental_rerun()

# kick off
if st.session_state.running:
    run_loop()
else:
    status.success("Done!")
    if st.session_state.engine is None:
        new_engine()
    redraw(0)