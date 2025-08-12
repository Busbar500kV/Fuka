# app.py
import json
from typing import Dict, Any

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use("dark_background")

import plotly.graph_objects as go

from sim_core import Engine, Config, FieldCfg

# ---------- helpers ----------

def build_engine_from_dict(cfg: Dict[str, Any]) -> Engine:
    # sources JSON
    raw = cfg.get("sources_json", "").strip()
    if raw:
        try:
            sources = json.loads(raw)
            if not isinstance(sources, list):
                raise ValueError("Sources JSON must be a list.")
        except Exception as e:
            st.error(f"Invalid Sources JSON: {e}")
            sources = FieldCfg().sources
    else:
        sources = FieldCfg().sources

    fcfg = FieldCfg(
        length=int(cfg.get("env_length", 512)),
        frames=int(cfg.get("frames", 5000)),
        noise_sigma=float(cfg.get("env_noise", 0.01)),
        sources=sources,
    )
    ecfg = Config(
        seed=int(cfg.get("seed", 0)),
        frames=int(cfg.get("frames", 5000)),
        space=int(cfg.get("space", 64)),
        k_flux=float(cfg.get("k_flux", 0.05)),
        k_motor=float(cfg.get("k_motor", 2.0)),
        decay=float(cfg.get("decay", 0.01)),
        diffuse=float(cfg.get("diffuse", 0.15)),
        env=fcfg,
    )
    return Engine(ecfg)

def energy_fig(hist):
    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=120)
    ax.plot(hist.t, hist.E_cell, label="E_cell")
    ax.plot(hist.t, hist.E_env,  label="E_env")
    ax.plot(hist.t, hist.E_flux, label="E_flux")
    ax.set_title("Energies (live)"); ax.set_xlabel("frame"); ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    return fig

def draw_combined_overlay(ph, env, S):
    # transpose to (space, frame)
    E = env.T; M = S.T
    # normalize separately
    En = (E - E.min()) / (E.ptp() + 1e-12)
    Mn = (M - M.min()) / (M.ptp() + 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Mn, colorscale="Inferno", showscale=True, opacity=0.72,
                             colorbar=dict(title="S", x=1.02)))
    fig.add_trace(go.Heatmap(z=En, colorscale="Viridis", showscale=True, opacity=0.55,
                             colorbar=dict(title="E", x=1.08)))
    fig.update_layout(template="plotly_dark", title="Combined: Env + Substrate (zoomable)",
                      xaxis_title="frame", yaxis_title="space",
                      margin=dict(l=40, r=90, t=40, b=40), height=420)
    with ph:
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- UI ----------

st.set_page_config(page_title="Fuka – Free‑Energy Gradient Simulation (Live)",
                   layout="wide", page_icon="🧪")
st.title("Fuka – Free‑Energy Gradient Simulation (Live)")

with st.sidebar:
    st.subheader("Controls")
    seed   = st.number_input("Seed",   0, 10_000, value=0, step=1)
    frames = st.number_input("Frames", 100, 50_000, value=5000, step=100)
    space  = st.number_input("Space (cells)", 16, 512, value=64, step=1)

    k_flux  = st.number_input("k_flux (boundary flux)", 0.0, 2.0, value=0.05, step=0.01)
    k_motor = st.number_input("k_motor (motor explore)", 0.0, 5.0, value=2.0,  step=0.10)
    decay   = st.number_input("decay",   0.0, 0.5, value=0.01, step=0.001)
    diffuse = st.number_input("diffuse", 0.0, 1.0, value=0.15, step=0.01)

    st.markdown("---")
    env_length = st.number_input("Env length", 32, 2048, value=512, step=1)
    env_noise  = st.number_input("Env noise σ", 0.0, 1.0, value=0.01, step=0.01)

    st.caption("Sources JSON (list). Example:\n"
               "`[{\"kind\":\"moving_peak\",\"amp\":1.0,\"speed\":0.0,\"width\":5,\"start\":340}]`")
    sources_json = st.text_area("Sources JSON", height=120, value=json.dumps([
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0,  "width": 5.0, "start": 340}
    ], indent=2))

    st.markdown("---")
    live  = st.checkbox("Live streaming", value=True)
    chunk = st.slider("Refresh chunk (frames)", 10, 400, 150, step=10)

    run_btn = st.button("Run / Rerun", use_container_width=True)

status_ph = st.empty()
energy_ph = st.empty()
combo_ph  = st.empty()

cfg: Dict[str, Any] = dict(
    seed=seed, frames=frames, space=space,
    k_flux=k_flux, k_motor=k_motor, decay=decay, diffuse=diffuse,
    env_length=env_length, env_noise=env_noise, sources_json=sources_json
)

def run():
    engine = build_engine_from_dict(cfg)

    # initial draw
    energy_ph.pyplot(energy_fig(engine.hist), clear_figure=True, use_container_width=True)
    draw_combined_overlay(combo_ph, engine.env, engine.S)

    if not live:
        engine.run(progress_cb=lambda _t: None)
        energy_ph.pyplot(energy_fig(engine.hist), clear_figure=True, use_container_width=True)
        draw_combined_overlay(combo_ph, engine.env, engine.S)
        status_ph.success("Done!")
        return

    status_ph.info("Starting…")
    last = -10_000

    def cb(t: int):
        nonlocal last
        # live energy every step
        energy_ph.pyplot(energy_fig(engine.hist), clear_figure=True, use_container_width=True)
        # combined heatmap every chunk
        if t - last >= int(chunk) or t == engine.cfg.frames - 1:
            last = t
            draw_combined_overlay(combo_ph, engine.env, engine.S)
            status_ph.info(f"Running… t={t+1}/{engine.cfg.frames}")

    try:
        engine.run(progress_cb=cb)
    except TypeError:
        engine.run(progress_cb=cb)

    energy_ph.pyplot(energy_fig(engine.hist), clear_figure=True, use_container_width=True)
    draw_combined_overlay(combo_ph, engine.env, engine.S)
    status_ph.success("Done!")

if run_btn:
    run()