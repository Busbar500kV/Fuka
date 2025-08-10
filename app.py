# app.py
# Streamlit UI with strong diagnostics + live streaming plots
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- Attempt import and show diagnostics early ---
st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

import sim_core as sc  # whatever Streamlit actually loads

# Pretty diagnostics
with st.sidebar:
    st.subheader("Module diagnostics")
    st.caption(f"sim_core file path: `{getattr(sc, '__file__', '<none>')}`")
    exports = sorted([k for k in dir(sc) if not k.startswith("_")])[:100]
    st.caption("Available symbols: " + ", ".join(exports))

    if st.button("Force reload sim_core"):
        importlib.invalidate_caches()
        sc = importlib.reload(sc)
        st.success("Reloaded sim_core. Press Rerun (top-right).")

# Import required names (or show a friendly message & stop)
MISSING = []
try:
    from sim_core import default_config, make_engine, Engine  # type: ignore
except Exception:
    # discover which ones are missing
    try:
        from sim_core import default_config  # type: ignore
    except Exception:
        MISSING.append("default_config")
    try:
        from sim_core import make_engine  # type: ignore
    except Exception:
        MISSING.append("make_engine")
    try:
        from sim_core import Engine  # type: ignore
    except Exception:
        MISSING.append("Engine")

if MISSING:
    st.error(
        "sim_core was imported but does not expose the required names.\n\n"
        f"Missing: {', '.join(MISSING)}\n\n"
        "Did you push the latest `sim_core.py` and restart the app?"
    )
    # show head of the file to help verify
    try:
        head = "\n".join(Path(sc.__file__).read_text(encoding="utf-8").splitlines()[:80])
        with st.expander("sim_core.py (first 80 lines)"):
            st.code(head, language="python")
    except Exception:
        pass
    st.stop()

# -------------------------
# UI controls
# -------------------------
st.title("Fuka: Free‑Energy Simulation")

cfg: Dict = default_config()

with st.sidebar:
    st.subheader("Run settings")
    cfg["frames"] = int(
        st.number_input("Frames", min_value=200, max_value=20000, value=cfg["frames"], step=200)
    )
    cfg["space"] = int(
        st.number_input("Space (1D sites)", min_value=32, max_value=1024, value=cfg["space"], step=32)
    )
    cfg["n_conns"] = int(st.number_input("Connections", 3, 128, cfg["n_conns"], 1))
    cfg["seed"] = int(st.number_input("Seed", 0, 10_000, cfg["seed"], 1))
    st.divider()
    st.subheader("Environment")
    cfg["env_power"] = float(st.number_input("Env power", 0.0, 10.0, cfg["env_power"], 0.1))
    cfg["env_hotspots"] = int(st.number_input("Env rays", 1, 32, cfg["env_hotspots"], 1))
    cfg["env_decay"] = float(st.slider("Env decay", 0.90, 0.999, float(cfg["env_decay"]), 0.001))
    st.divider()
    st.subheader("Energy & Costs")
    cfg["pool_init"] = float(st.number_input("Initial energy", -1_000.0, 1_000.0, cfg["pool_init"], 10.0))
    cfg["act_cost"] = float(st.number_input("Activation cost", 0.0, 0.2, cfg["act_cost"], 0.005))
    cfg["main_cost"] = float(st.number_input("Maintenance cost", 0.0, 0.01, cfg["main_cost"], 0.0005))
    st.divider()
    st.subheader("Drive weights")
    cfg["w_direct"] = float(st.slider("Direct weight", 0.0, 2.0, float(cfg["w_direct"]), 0.05))
    cfg["w_indirect"] = float(st.slider("Indirect weight", 0.0, 2.0, float(cfg["w_indirect"]), 0.05))
    cfg["w_motor"] = float(st.slider("Motor weight", 0.0, 2.0, float(cfg["w_motor"]), 0.05))

    live_mode = st.checkbox("Live stream while running", value=True)
    chunk = int(st.number_input("Stream every N steps", 5, 200, 20, 5))

run_btn = st.button("Run simulation", type="primary")

# -------------------------
# Runner
# -------------------------
if run_btn:
    st.success("Starting…")
    engine = make_engine(cfg)

    # layout
    col1, col2 = st.columns([1, 1])
    energy_chart = col1.line_chart({"energy": [engine.energy[0]]})
    env_fig = col2.empty()
    subs_fig = col1.empty()
    act_fig = col2.empty()
    prog = st.progress(0, text="Running…")

    def redraw(t_now: int):
        if not live_mode:
            return
        T = t_now + 1
        # energy
        energy_chart.add_rows({"energy": engine.energy[:T]})

        # env heatmap
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        im1 = ax1.imshow(engine.env[:T].T, aspect="auto", origin="lower")
        ax1.set_title("Environment")
        ax1.set_xlabel("frame"); ax1.set_ylabel("space")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        env_fig.pyplot(fig1); plt.close(fig1)

        # substrate heatmap
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        im2 = ax2.imshow(engine.subs[:T].T, aspect="auto", origin="lower", cmap="magma")
        ax2.set_title("Substrate")
        ax2.set_xlabel("frame"); ax2.set_ylabel("space")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        subs_fig.pyplot(fig2); plt.close(fig2)

        # activations (connections × time)
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        im3 = ax3.imshow(engine.act[:T].T, aspect="auto", origin="lower", cmap="viridis")
        ax3.set_title("Connection activations")
        ax3.set_xlabel("frame"); ax3.set_ylabel("connection idx")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        act_fig.pyplot(fig3); plt.close(fig3)

        prog.progress(min(1.0, T / cfg["frames"]), text=f"Running… {T}/{cfg['frames']}")

    # run with streaming
    if live_mode:
        last = 0
        def cb(t):
            nonlocal last
            if t - last >= chunk or t == cfg["frames"] - 1:
                last = t
                redraw(t)
        engine.run(progress_cb=cb)
        redraw(cfg["frames"] - 1)
    else:
        engine.run(progress_cb=None)
        redraw(cfg["frames"] - 1)

    st.success("Done!")