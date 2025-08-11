# app.py
# Streamlit UI for live simulation with adjustable environment sources.

from __future__ import annotations
import json
from copy import deepcopy
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sim_core import Engine, default_config, FieldCfg, Config


st.set_page_config(page_title="Fuka – Free‑Energy Gradient (Live)", layout="wide")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.title("Controls")

# session config dict
if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()
cfg: Dict = st.session_state.cfg

def clamp(v, lo, hi): return int(max(lo, min(hi, v)))

# basic knobs
cfg["seed"]   = st.sidebar.number_input("Seed", 0, 1_000_000, int(cfg.get("seed", 0)), 1)
cfg["frames"] = clamp(st.sidebar.number_input("Frames", 10000, 20000, int(cfg.get("frames", 1600)), 100), 200, 5000)
cfg["space"]  = clamp(st.sidebar.number_input("Space (cells)", 32, 512, int(cfg.get("space", 192)), 16), 32, 512)

# dynamics
cols = st.sidebar.columns(2)
cfg["k_flux"]  = float(cols[0].number_input("k_flux", 0.0, 1.0, float(cfg.get("k_flux", 0.04)), 0.01))
cfg["k_motor"] = float(cols[1].number_input("k_motor", 0.0, 3.0, float(cfg.get("k_motor", 2.0)), 0.01))
cols = st.sidebar.columns(2)
cfg["diffuse"] = float(cols[0].number_input("diffuse", 0.0, 1.0, float(cfg.get("diffuse", 0.06)), 0.01))
cfg["decay"]   = float(cols[1].number_input("decay", 0.0, 0.5, float(cfg.get("decay", 0.01)), 0.002))

# environment / sources
env = cfg.setdefault("env", {})
env["length"]     = clamp(st.sidebar.number_input("Env length", 32, 512, int(env.get("length", cfg["space"])), 16), 32, 512)
env["frames"]     = cfg["frames"]
env["noise_sigma"]= float(st.sidebar.number_input("Env noise σ", 0.0, 0.5, float(env.get("noise_sigma", 0.01)), 0.01))

# sources JSON
default_sources = FieldCfg().sources
if "sources" not in env:
    env["sources"] = deepcopy(default_sources)

st.sidebar.markdown("**Sources JSON** (list)")
src_text = st.sidebar.text_area("Edit env sources",
                                value=json.dumps(env["sources"], indent=2),
                                height=240)
try:
    env["sources"] = json.loads(src_text)
    st.sidebar.success("Sources OK")
except Exception as e:
    st.sidebar.error(f"Invalid JSON for sources: {e}")

# run controls
col_run = st.sidebar.container()
live_mode = col_run.checkbox("Live streaming", True)
chunk = col_run.slider("Refresh chunk (frames)", 10, 200, 50, 10)
run_btn = col_run.button("Run / Restart", type="primary")


# -----------------------
# Main area
# -----------------------
st.title("Fuka – Free‑Energy Gradient Simulation (Live)")
status = st.empty()
fig_placeholder = st.empty()
heat_cols = st.columns(2)
env_ph = heat_cols[0].empty()
S_ph   = heat_cols[1].empty()

def plot_heat(ax, M, title: str):
    ax.imshow(M.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("space")

def redraw(t: int, eng: Engine):
    # line chart of energies
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(eng.hist.t, eng.hist.E_cell, label="E_cell")
    ax.plot(eng.hist.t, eng.hist.E_env,  label="E_env")
    ax.plot(eng.hist.t, eng.hist.E_flux, label="E_flux")
    ax.set_xlim(0, eng.T-1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig_placeholder.pyplot(fig)
    plt.close(fig)

    # env & substrate heatmaps
    f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))
    plot_heat(ax1, eng.env[:t+1], "Environment E(t,x)")
    plot_heat(ax2, eng.S[:t+1],   "Substrate S(t,x)")
    env_ph.pyplot(f2)
    plt.close(f2)

if run_btn:
    status.info("Starting…")
    # Build Engine from current cfg
    cfg_dict = deepcopy(cfg)
    cfg_dict["env"]["frames"] = cfg_dict["frames"]  # keep in sync
    engine = Engine(Config(
        seed=cfg_dict["seed"],
        frames=cfg_dict["frames"],
        space=cfg_dict["space"],
        n_init=cfg_dict.get("n_init", 9),
        k_flux=cfg_dict["k_flux"],
        k_motor=cfg_dict["k_motor"],
        decay=cfg_dict["decay"],
        diffuse=cfg_dict["diffuse"],
        env=FieldCfg(
            length=cfg_dict["env"]["length"],
            frames=cfg_dict["frames"],
            noise_sigma=cfg_dict["env"]["noise_sigma"],
            sources=cfg_dict["env"]["sources"],
        ),
    ))

    if live_mode:
        last = [0]  # mutable to capture inside callback

        def cb(t: int):
            if (t - last[0] >= chunk) or (t == engine.T - 1):
                last[0] = t
                redraw(t, engine)

        engine.run(progress_cb=cb)
        redraw(engine.T - 1, engine)
    else:
        engine.run()
        redraw(engine.T - 1, engine)

    status.success("Done!")