from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import importlib

# -----------------------------------------------------------------------------
# Import module ONLY; do not import names directly
# -----------------------------------------------------------------------------
import sim_core as sc  # this must succeed, or the app cannot run

# Resolve symbols with safe fallbacks + diagnostics
default_config = getattr(sc, "default_config", None)
Engine         = getattr(sc, "Engine", None)
make_engine    = getattr(sc, "make_engine", None)

# If make_engine is missing but Engine exists, provide a wrapper
if make_engine is None and Engine is not None:
    def make_engine(cfg):  # type: ignore
        return Engine(cfg)

# If anything critical is missing, surface helpful info and stop
missing = []
if default_config is None: missing.append("default_config")
if make_engine is None and Engine is None: missing.append("Engine/make_engine")

if missing:
    st.error("`sim_core` was imported but does not expose the required names.")
    st.write("Imported from:", getattr(sc, "__file__", "(no __file__)"))
    st.write("Available symbols:", [n for n in dir(sc) if not n.startswith("_")])
    st.stop()

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")
st.title("Fuka — Free‑Energy Simulation")

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Run Settings")

    # Start with defaults from sim_core.py but allow edits
    cfg = dict(default_config)

    frames = st.number_input("Frames per run", min_value=100, max_value=20000, value=cfg.get("frames", 1600), step=200)
    space  = st.number_input("Space (N_s)",     min_value=8,   max_value=2048,  value=cfg.get("space", 192), step=8)
    seed   = st.number_input("Seed",            min_value=0,   max_value=1_000_000, value=cfg.get("seed", 0), step=1)

    # Environment sliders
    env_E0  = st.slider("Env energy E0", 0.0,  10.0, float(cfg.get("env_E0", 2.0)), 0.1)
    env_var = st.slider("Env variance",  0.0,   5.0, float(cfg.get("env_var", 0.5)), 0.1)

    # Mutation / growth
    grow_budget = st.number_input("Grow budget per epoch", 0, 200, cfg.get("grow_budget", 10), 1)
    prune_tau   = st.slider("Prune time constant", 10, 5000, int(cfg.get("prune_tau", 800)), 10)

    # Apply to cfg
    cfg.update(
        frames=int(frames),
        space=int(space),
        seed=int(seed),
        env_E0=float(env_E0),
        env_var=float(env_var),
        grow_budget=int(grow_budget),
        prune_tau=int(prune_tau),
    )

    run_btn = st.button("Run")

# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
status = st.empty()
col1, col2 = st.columns(2)
chart_energy = col1.empty()
chart_counts = col2.empty()
table_final  = st.empty()

if run_btn:
    status.info("Initializing engine…")
    eng = make_engine(cfg)

    E_hist, S_hist = [], []
    n_sense_hist, n_internal_hist, n_motor_hist = [], [], []

    # Run the requested frames
    for f in range(cfg["frames"]):
        out = eng.step()  # one tick

        # Track simple diagnostics
        E_hist.append(out["E"])
        S_hist.append(out["S"])
        n_sense_hist.append(out["n_sense"])
        n_internal_hist.append(out["n_internal"])
        n_motor_hist.append(out["n_motor"])

        # Update charts intermittently for responsiveness
        if (f % 50 == 0) or (f == cfg["frames"] - 1):
            # Energy chart
            fig1, ax1 = plt.subplots()
            ax1.plot(E_hist, label="Free Energy")
            ax1.plot(S_hist, label="Entropy")
            ax1.set_title("Energy / Entropy")
            ax1.legend()
            chart_energy.pyplot(fig1)
            plt.close(fig1)

            # Counts chart
            fig2, ax2 = plt.subplots()
            ax2.plot(n_sense_hist, label="SENSE")
            ax2.plot(n_internal_hist, label="INTERNAL")
            ax2.plot(n_motor_hist, label="MOTOR")
            ax2.set_title("Connection Counts")
            ax2.legend()
            chart_counts.pyplot(fig2)
            plt.close(fig2)

            status.info(f"Running… frame {f+1}/{cfg['frames']}")

    status.success("Done!")

    # Final connection table
    df = eng.summary_table()
    table_final.dataframe(df)

# Footer diagnostics
with st.expander("Diagnostics"):
    st.write("sim_core file:", getattr(sc, "__file__", "(no __file__)"))
    st.write("Exports present:", [n for n in dir(sc) if not n.startswith("_")])