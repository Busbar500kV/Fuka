"""
app.py — Streamlit UI with strong import diagnostics and a live loop.

- Shows exactly what `sim_core` module your app has imported.
- Lets you configure parameters, initialize, and run continuously.
- Renders live charts and a connection table.

Tip: After pasting both files, commit, then "Restart" the app from Streamlit Cloud.
"""

from __future__ import annotations

import time
import inspect
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Import the module two ways to show what’s actually loaded ---
import sim_core as sc
from sim_core import default_config, make_engine, Engine  # will raise ImportError if missing


# -------------------------
# Page config & style
# -------------------------
st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")


# -------------------------
# Diagnostics pane
# -------------------------
with st.expander("🔍 Diagnostics (click to open) — what did Python import?"):
    st.write("**sim_core module file:**", getattr(sc, "__file__", "(no __file__)"))
    st.write("**Exported symbols:**", sorted([x for x in dir(sc) if not x.startswith("_")]))

    try:
        d = sc.diag_info()
    except Exception as e:
        d = {"diag_error": repr(e)}
    st.json(d)

    # show hashes of app + engine sources so we can see they match the current repo
    try:
        src_engine = inspect.getsource(Engine)
        st.write("Engine source hash:", hashlib.sha256(src_engine.encode("utf-8")).hexdigest()[:12])
    except Exception as e:
        st.write("Engine source hash: (error)", repr(e))

    try:
        src_factory = inspect.getsource(make_engine)
        st.write("make_engine source hash:", hashlib.sha256(src_factory.encode("utf-8")).hexdigest()[:12])
    except Exception as e:
        st.write("make_engine source hash: (error)", repr(e))


# -------------------------
# Helpers
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ensure_engine():
    if "engine" not in st.session_state or st.session_state.engine is None:
        st.session_state.engine = make_engine(default_config)


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Work off a copy of defaults
    cfg = dict(default_config)

    # Safer number_inputs (min_value ≤ value ≤ max_value)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=int(cfg["seed"]), step=1)
    frames = st.number_input("Frames per run (for manual step/run)", min_value=10, max_value=100_000, value=int(cfg["frames"]), step=100)
    space = st.number_input("Space (label)", min_value=16, max_value=4096, value=int(cfg["space"]), step=16)

    env_free_energy = st.slider("Environment free energy (baseline)", 0.0, 5.0, float(cfg["env_free_energy"]), 0.05)
    noise_env = st.slider("Environment noise", 0.0, 1.0, float(cfg["noise_env"]), 0.01)

    dt = st.slider("Δt", 0.01, 1.0, float(cfg["dt"]), 0.01)
    harvest_gain = st.slider("Harvest gain", 0.0, 1.0, float(cfg["harvest_gain"]), 0.01)
    internal_gain = st.slider("Internal gain", 0.0, 1.0, float(cfg["internal_gain"]), 0.01)
    cost_activity = st.slider("Cost activity", 0.0, 0.2, float(cfg["cost_activity"]), 0.005)
    cost_maintenance = st.slider("Cost maintenance", 0.0, 0.2, float(cfg["cost_maintenance"]), 0.005)

    # build config dict
    cfg.update({
        "seed": int(seed),
        "frames": int(frames),
        "space": int(space),
        "env_free_energy": float(env_free_energy),
        "noise_env": float(noise_env),
        "dt": float(dt),
        "harvest_gain": float(harvest_gain),
        "internal_gain": float(internal_gain),
        "cost_activity": float(cost_activity),
        "cost_maintenance": float(cost_maintenance),
    })

    # Buttons
    colA, colB = st.columns(2)
    with colA:
        if st.button("Initialize / Rebuild Engine", use_container_width=True):
            st.session_state.engine = make_engine(cfg)
            st.toast("Engine (re)initialized.", icon="✅")
    with colB:
        if st.button("Reset State (same config)", use_container_width=True):
            ensure_engine()
            st.session_state.engine.reset(seed=int(cfg["seed"]))
            st.toast("Engine reset.", icon="♻️")

    st.divider()
    autorun = st.checkbox("Continuous run (auto)", value=False)
    run_rate = st.slider("Steps per tick (when auto)", 1, 200, 25, 1)
    sleep_ms = st.slider("UI sleep (ms)", 0, 500, 50, 10)


# -------------------------
# Main viewport
# -------------------------
ensure_engine()
eng: Engine = st.session_state.engine

st.title("Fuka — Free‑Energy Gradient Simulation (diagnostic build)")

topA, topB, topC = st.columns([2, 1, 1])
with topA:
    st.caption("**Status**")
    st.json(eng.summary())
with topB:
    if st.button("Step × 1"):
        next(eng.run(1, yield_every=1))
        st.rerun()
    if st.button("Run × N (from sidebar 'Frames')"):
        # Run a short burst and then re-render once to keep UI snappy
        for _ in eng.run(int(cfg["frames"]), yield_every=max(1, int(cfg["frames"]) // 10)):
            pass
        st.rerun()
with topC:
    st.write("—")
    st.write("**Time**:", eng.t)
    st.write("**Connections**:", len(eng.conns))

# Live charts
chartA, chartB = st.columns(2)
with chartA:
    st.subheader("Energy over time")
    fig, ax = plt.subplots()
    if eng.time:
        ax.plot(eng.time, eng.E_total, label="E_internal")
        ax.plot(eng.time, eng.E_env, label="E_env (boundary)")
        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("Energy")
    st.pyplot(fig, clear_figure=True)

with chartB:
    st.subheader("Harvest vs Costs")
    fig2, ax2 = plt.subplots()
    if eng.time:
        ax2.plot(eng.time, eng.P_harvest, label="Harvest")
        ax2.plot(eng.time, eng.Costs, label="Costs")
        ax2.legend()
        ax2.set_xlabel("t")
        ax2.set_ylabel("Power")
    st.pyplot(fig2, clear_figure=True)

st.subheader("Connections")
df = pd.DataFrame(eng.connections_table())
st.dataframe(df, use_container_width=True, height=240)

# Continuous loop
if autorun:
    steps = int(run_rate)
    # do small chunk then sleep briefly to keep the Streamlit event loop happy
    for _ in eng.run(steps, yield_every=steps):
        pass
    if int(sleep_ms) > 0:
        time.sleep(int(sleep_ms) / 1000.0)
    # Rerun the script to update charts
    st.experimental_rerun()