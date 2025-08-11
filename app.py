import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sim_core import Engine, default_config, run_sim

st.set_page_config(page_title="Fuka: Free-Energy Simulation", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()

cfg = st.session_state.cfg

with st.sidebar:
    st.markdown("### Simulation Parameters")
    cfg["frames"] = st.number_input("Frames", 100, 20000, value=int(cfg.get("frames", 1600)), step=100)
    cfg["space"]  = st.number_input("Space (cells)", 8, 1024, value=int(cfg.get("space", 64)), step=1)
    cfg["seed"]   = st.number_input("Seed", 0, 99999, value=int(cfg.get("seed", 0)), step=1)
    cfg["k_flux"] = st.number_input("k_flux", 0.0, 5.0, value=float(cfg.get("k_flux", 0.05)), step=0.01)
    cfg["k_motor"]= st.number_input("k_motor", 0.0, 5.0, value=float(cfg.get("k_motor", 2.0)), step=0.01)
    cfg["k_noise"]= st.number_input("k_noise", 0.0, 5.0, value=float(cfg.get("k_noise", 0.0)), step=0.01)
    cfg["decay"]  = st.number_input("decay", 0.0, 1.0, value=float(cfg.get("decay", 0.01)), step=0.001)
    cfg["diffuse"]= st.number_input("diffuse", 0.0, 1.0, value=float(cfg.get("diffuse", 0.05)), step=0.001)
    cfg["band"]   = st.number_input("Boundary Band", 1, 64, value=int(cfg.get("band", 3)), step=1)

    st.markdown("### Environment JSON")
    env_json = st.text_area(
        "Edit environment config JSON",
        value=json.dumps(cfg.get("env", {}), indent=2),
        height=300
    )
    try:
        cfg["env"] = json.loads(env_json)
    except json.JSONDecodeError:
        st.error("Invalid JSON in environment config")

if st.button("Run Simulation"):
    hist, env, S = run_sim(cfg)

    fig_env = go.Figure(data=go.Heatmap(
        z=env,
        colorscale='Viridis'
    ))
    fig_env.update_layout(title="Environment Field (E)", xaxis_title="X", yaxis_title="Time")

    fig_S = go.Figure(data=go.Heatmap(
        z=S,
        colorscale='Inferno'
    ))
    fig_S.update_layout(title="Substrate (S)", xaxis_title="X", yaxis_title="Time")

    df_hist = pd.DataFrame({
        "t": hist.t,
        "E_cell": hist.E_cell,
        "E_env": hist.E_env,
        "E_flux": hist.E_flux
    })
    fig_hist = go.Figure()
    for col in ["E_cell", "E_env", "E_flux"]:
        fig_hist.add_trace(go.Scatter(x=df_hist["t"], y=df_hist[col], mode="lines", name=col))
    fig_hist.update_layout(title="Energy Metrics over Time", xaxis_title="Time", yaxis_title="Value")

    st.plotly_chart(fig_env, use_container_width=True)
    st.plotly_chart(fig_S, use_container_width=True)
    st.plotly_chart(fig_hist, use_container_width=True)