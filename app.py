# app.py
import json
from typing import Dict, Any

import numpy as np
import streamlit as st

# Matplotlib for the energy line plot (now using dark theme)
import matplotlib.pyplot as plt
plt.style.use("dark_background")

# Plotly for the zoomable combined heatmap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sim_core import Engine, Config, FieldCfg  # physics unchanged


# ---------------------------
# Helpers
# ---------------------------

def build_engine_from_dict(cfg: Dict[str, Any]) -> Engine:
    """Build Engine from UI dict safely (keeps physics exactly as in sim_core)."""
    # Parse env sources JSON (list of dicts)
    sources = cfg.get("sources_json", "").strip()
    if sources:
        try:
            parsed = json.loads(sources)
            assert isinstance(parsed, list), "Sources JSON must be a list of dicts"
            sources_list = parsed
        except Exception as e:
            st.error(f"Invalid Sources JSON: {e}")
            sources_list = FieldCfg().sources
    else:
        sources_list = FieldCfg().sources

    fcfg = FieldCfg(
        length=int(cfg.get("env_length", 512)),
        frames=int(cfg.get("frames", 5000)),
        noise_sigma=float(cfg.get("env_noise", 0.01)),
        sources=sources_list,
    )

    ecfg = Config(
        seed=int(cfg.get("seed", 0)),
        frames=int(cfg.get("frames", 5000)),
        space=int(cfg.get("space", 64)),
        n_init=9,
        k_flux=float(cfg.get("k_flux", 0.05)),
        k_motor=float(cfg.get("k_motor", 2.0)),
        decay=float(cfg.get("decay", 0.01)),
        diffuse=float(cfg.get("diffuse", 0.15)),
        env=fcfg,
    )
    return Engine(ecfg)


def draw_energy(ax, hist):
    ax.clear()
    ax.plot(hist.t, hist.E_cell, label="E_cell")
    ax.plot(hist.t, hist.E_env,  label="E_env")
    ax.plot(hist.t, hist.E_flux, label="E_flux")
    ax.set_title("Energies (live)")
    ax.set_xlabel("frame")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def draw_combined_plotly(ph, env, S, mode="overlay"):
    """
    Zoomable combined view with Plotly.
    mode = "overlay" -> E and S on same axes, different colors, with opacity
    mode = "stacked" -> two synchronized panels
    """
    # env and substrate arrive as (T, X); Plotly Heatmap expects (y, x) = (space, time)
    E = env.T
    M = S.T

    # normalize separately for visual clarity
    En = (E - E.min()) / (E.ptp() + 1e-12)
    Mn = (M - M.min()) / (M.ptp() + 1e-12)

    if mode == "overlay":
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=Mn, colorscale="Inferno", showscale=True, opacity=0.72,
            colorbar=dict(title="S", x=1.02)
        ))
        fig.add_trace(go.Heatmap(
            z=En, colorscale="Viridis", showscale=True, opacity=0.55,
            colorbar=dict(title="E", x=1.08)
        ))
        fig.update_layout(
            template="plotly_dark",
            title="Combined (overlay: S=Inferno, E=Viridis)",
            xaxis_title="frame", yaxis_title="space",
            margin=dict(l=40, r=90, t=40, b=40), height=420
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Environment E(t,x)", "Substrate S(t,x)")
        )
        fig.add_trace(go.Heatmap(z=En, colorscale="Viridis", showscale=True,
                                 colorbar=dict(title="E")), row=1, col=1)
        fig.add_trace(go.Heatmap(z=Mn, colorscale="Inferno", showscale=True,
                                 colorbar=dict(title="S")), row=2, col=1)
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="frame", xaxis2_title="frame",
            yaxis_title="space", yaxis2_title="space",
            margin=dict(l=40, r=40, t=60, b=40), height=700
        )

    with ph:
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------------------------
# UI
# ---------------------------

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
    env_len   = st.number_input("Env length", 32, 2048, value=512, step=1)
    env_noise = st.number_input("Env noise σ", 0.0, 1.0, value=0.01, step=0.01)

    st.caption("Sources JSON (list of dicts). Example:\n"
               "`[{\"kind\":\"moving_peak\",\"amp\":1.0,\"speed\":0.0,\"width\":5,\"start\":340}]`")
    sources_json = st.text_area("Sources JSON", height=120, value=json.dumps([
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0,  "width": 5.0, "start": 340}
    ], indent=2))

    st.markdown("---")
    live = st.checkbox("Live streaming", value=True)
    chunk = st.slider("Refresh chunk (frames)", 10, 400, 150, step=10)
    mode = st.selectbox("Combined view", ["overlay", "stacked"], index=0)

    run_btn = st.button("Run / Rerun", use_container_width=True)

# placeholders
status_ph  = st.empty()
energy_ph  = st.empty()
combo_ph   = st.empty()

# persistent figure for energy (matplotlib)
fig_energy, ax_energy = plt.subplots(figsize=(6.5, 2.8), dpi=120)

# Pack the UI dict
user_cfg: Dict[str, Any] = dict(
    seed=seed, frames=frames, space=space,
    k_flux=k_flux, k_motor=k_motor, decay=decay, diffuse=diffuse,
    env_length=env_len, env_noise=env_noise, sources_json=sources_json
)


def run_once():
    engine = build_engine_from_dict(user_cfg)

    # initial draw
    draw_energy(ax_energy, engine.hist)
    with energy_ph:
        st.pyplot(fig_energy, clear_figure=False, use_container_width=True)

    draw_combined_plotly(combo_ph, engine.env, engine.S, mode=mode)

    if not live:
        # one-shot run
        def cb(_t): pass
        engine.run(progress_cb=cb)
        draw_energy(ax_energy, engine.hist)
        energy_ph.pyplot(fig_energy, clear_figure=False, use_container_width=True)
        draw_combined_plotly(combo_ph, engine.env, engine.S, mode=mode)
        status_ph.success("Done!")
        return

    # live mode
    status_ph.info("Starting…")

    last_draw = -999999

    def cb(t: int):
        nonlocal last_draw
        # update energy line
        draw_energy(ax_energy, engine.hist)
        energy_ph.pyplot(fig_energy, clear_figure=False, use_container_width=True)

        if t - last_draw >= chunk or t == engine.cfg.frames - 1:
            last_draw = t
            draw_combined_plotly(combo_ph, engine.env, engine.S, mode=mode)
            status_ph.info(f"Running… t={t+1}/{engine.cfg.frames}")

    # run engine (compatible with older Engine.run signatures)
    try:
        engine.run(progress_cb=cb)  # your Engine.run has only progress_cb
    except TypeError:
        # If your Engine.run has snapshot_every, you can switch to it here
        engine.run(progress_cb=cb)

    # final refresh
    draw_energy(ax_energy, engine.hist)
    energy_ph.pyplot(fig_energy, clear_figure=False, use_container_width=True)
    draw_combined_plotly(combo_ph, engine.env, engine.S, mode=mode)
    status_ph.success("Done!")


if run_btn:
    run_once()