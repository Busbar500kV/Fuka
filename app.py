import os
# Prevent inotify "instance limit reached" error
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core.simulation import run_simulation, default_config
from core.plot import (
    draw_energy_timeseries,
    draw_overlay_last_frame,
    draw_heatmap_full
)

st.set_option("server.fileWatcherType", "none")
st.set_option("server.runOnSave", False)

st.set_page_config(
    page_title="Fuka: Free-Energy Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Fuka: Free-Energy Simulation")

# Sidebar configuration
st.sidebar.header("Simulation Parameters")
cfg = default_config()

# Simulation controls
cfg["seed"] = st.sidebar.number_input("Random Seed", value=cfg["seed"])
cfg["frames"] = st.sidebar.number_input("Frames", value=cfg["frames"])
cfg["space"] = st.sidebar.number_input("Space Size", value=cfg["space"])
cfg["k_flux"] = st.sidebar.number_input("k_flux", value=cfg["k_flux"], step=0.01)
cfg["k_motor"] = st.sidebar.number_input("k_motor", value=cfg["k_motor"], step=0.01)
cfg["k_noise"] = st.sidebar.number_input("k_noise", value=cfg["k_noise"], step=0.01)
cfg["source_pos"] = st.sidebar.number_input("Source Position", value=cfg["source_pos"])
cfg["source_val"] = st.sidebar.number_input("Source Value", value=cfg["source_val"])

# Run button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        hist, env, substrate, e_cell, e_env, e_flux = run_simulation(cfg)

        # Plots
        st.subheader("Energy Time Series")
        draw_energy_timeseries(st, e_cell, e_env, e_flux)

        st.subheader("Environment + Substrate Overlay (Last Frame)")
        draw_overlay_last_frame(st, env[-1], substrate[-1])

        st.subheader("Full Heatmap E(t,x) + S(t,x)")
        draw_heatmap_full(st, env, substrate)