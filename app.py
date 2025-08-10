import streamlit as st
from sim_core import run_sim, default_config

st.set_page_config(page_title="Energy-Gradient Simulation", layout="wide")

cfg = default_config()
# Expose a few top-level params as controls:
cfg["frames"] = st.sidebar.slider("Frames", 200, 10000, cfg["frames"], step=100)
cfg["free_energy_env"] = st.sidebar.slider("Env free energy", 0.0, 10.0, cfg["free_energy_env"], 0.1)
cfg["N_init"] = st.sidebar.slider("Initial connections", 1, 64, cfg["N_init"], 1)
cfg["seed"] = st.sidebar.number_input("Seed", value=cfg["seed"], step=1)

if st.button("Run / Continue"):
    state = st.session_state.get("sim_state")
    state = run_sim(cfg, prev_state=state)
    st.session_state["sim_state"] = state

# Render outputs
state = st.session_state.get("sim_state")
if state:
    st.write("Energy:", state["E"])
    st.write("Population:", state["N"])
    st.line_chart(state["energy_trace"])
    st.dataframe(state["final_table"])
    st.download_button("Download final table (CSV)",
                       data=state["final_csv"],
                       file_name="final_connections.csv",
                       mime="text/csv")
else:
    st.info("Click **Run / Continue** to start the simulation.")