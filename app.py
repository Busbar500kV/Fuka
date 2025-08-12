# app.py — Streamlit UI for live Fuka simulation
# Assumes sim_core.py exports: Engine, Config, FieldCfg, default_config

import json
import time
from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---- import from your core ----
from sim_core import Engine, Config, FieldCfg, default_config


st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

# ---------- helpers ----------

def build_engine_from_dict(cfg: Dict) -> Engine:
    """Translate a plain dict (from UI) into an Engine."""
    env_cfg = cfg.get("env", {})
    # parse sources (JSON text box may pass a string)
    src = env_cfg.get("sources", FieldCfg().sources)
    if isinstance(src, str):
        try:
            src = json.loads(src)
        except Exception:
            src = FieldCfg().sources

    fcfg = FieldCfg(
        length=int(env_cfg.get("length", 512)),
        frames=int(env_cfg.get("frames", cfg.get("frames", 5000))),
        noise_sigma=float(env_cfg.get("noise_sigma", 0.01)),
        sources=src,
    )
    ecfg = Config(
        seed=int(cfg.get("seed", 0)),
        frames=int(cfg.get("frames", 5000)),
        space=int(cfg.get("space", 64)),
        n_init=int(cfg.get("n_init", 9)),
        k_flux=float(cfg.get("k_flux", 0.05)),
        k_motor=float(cfg.get("k_motor", 2.0)),
        decay=float(cfg.get("decay", 0.01)),
        diffuse=float(cfg.get("diffuse", 0.05)),
        env=fcfg,
    )
    return Engine(ecfg)


def normalize_01(A: np.ndarray):
    a_min, a_max = float(np.min(A)), float(np.max(A))
    den = (a_max - a_min) if (a_max > a_min) else 1.0
    return (A - a_min) / den


def draw_energy(ph, hist):
    t = np.asarray(hist.t, dtype=float)
    if t.size == 0:
        with ph:
            st.empty()
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hist.E_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=hist.E_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=hist.E_flux, name="E_flux", mode="lines"))
    fig.update_layout(template="plotly_dark",
                      title="Energies over time",
                      xaxis_title="frame", yaxis_title="energy",
                      height=300, margin=dict(l=40, r=20, t=40, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    with ph:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def draw_combined_heatmap(ph, env: np.ndarray, S: np.ndarray):
    """Single zoomable plot combining Env and Substrate as two semi-transparent heatmaps."""
    if env is None or S is None or env.size == 0 or S.size == 0:
        with ph:
            st.empty()
        return

    # We want axes: x=frame, y=space. Our arrays are (T, X) so transpose to (X, T).
    E = np.asarray(env).T
    M = np.asarray(S).T

    En = normalize_01(E)
    Mn = normalize_01(M)

    fig = go.Figure()
    # Substrate first (so env overlays a bit)
    fig.add_trace(go.Heatmap(
        z=Mn, colorscale="Inferno", showscale=True, opacity=0.70,
        colorbar=dict(title="Substrate S", x=1.02)))
    # Environment over it
    fig.add_trace(go.Heatmap(
        z=En, colorscale="Viridis", showscale=True, opacity=0.55,
        colorbar=dict(title="Environment E", x=1.10)))

    fig.update_layout(
        template="plotly_dark",
        title="Combined (zoomable): Environment + Substrate",
        xaxis_title="frame",
        yaxis_title="space",
        height=420,
        margin=dict(l=40, r=120, t=40, b=40)
    )

    with ph:
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------- UI: sidebar knobs ----------
if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()  # plain dict from your core

cfg = st.session_state.cfg

with st.sidebar:
    st.markdown("## Run Controls")

    # core timing / sizes
    frames = st.number_input("Frames", min_value=200, max_value=20000, value=int(cfg.get("frames", 5000)), step=200)
    space  = st.number_input("Substrate size (space)", min_value=32, max_value=1024, value=int(cfg.get("space", 64)), step=32)
    seed   = st.number_input("Seed", min_value=0, max_value=10_000, value=int(cfg.get("seed", 0)), step=1)

    st.markdown("### Physics")
    k_flux  = st.slider("k_flux (boundary pump)", 0.0, 1.0, float(cfg.get("k_flux", 0.05)), 0.01)
    k_motor = st.slider("k_motor (random motor at boundary)", 0.0, 5.0, float(cfg.get("k_motor", 2.0)), 0.05)
    decay   = st.slider("decay (local loss)", 0.0, 0.2, float(cfg.get("decay", 0.01)), 0.005)
    diffuse = st.slider("diffuse (local mixing)", 0.0, 0.5, float(cfg.get("diffuse", 0.05)), 0.01)

    st.markdown("### Environment")
    env_len   = st.number_input("env.length", min_value=64, max_value=4096, value=int(cfg.get("env", {}).get("length", 512)), step=64)
    env_noise = st.slider("env.noise_sigma", 0.0, 0.5, float(cfg.get("env", {}).get("noise_sigma", 0.01)), 0.01)

    st.caption("Edit sources JSON (moving peaks). Example:\n"
               """[{"kind":"moving_peak","amp":1.0,"speed":0.10,"width":4.0,"start":24}]""")
    sources_text = st.text_area(
        "env.sources (JSON list)",
        value=json.dumps(cfg.get("env", {}).get("sources", FieldCfg().sources), indent=2),
        height=180
    )

    st.markdown("### Live update")
    chunk = st.slider("UI update chunk (frames)", 20, 500, 150, 10)

    run_btn = st.button("Run", type="primary")
    stop_btn = st.button("Stop")


# ---------- placeholders (single instances, repeatedly updated) ----------
col_top = st.columns([1, 1])
energy_ph = col_top[0].empty()
combo_ph  = col_top[1].empty()

log_ph = st.expander("Run log", expanded=False)
log_box = log_ph.empty()

# simple stop flag
if "stop" not in st.session_state:
    st.session_state.stop = False

if stop_btn:
    st.session_state.stop = True

def run():
    # capture user config back into dict
    cfg["frames"] = int(frames)
    cfg["space"]  = int(space)
    cfg["seed"]   = int(seed)
    cfg["k_flux"]  = float(k_flux)
    cfg["k_motor"] = float(k_motor)
    cfg["decay"]   = float(decay)
    cfg["diffuse"] = float(diffuse)
    cfg.setdefault("env", {})
    cfg["env"]["length"]      = int(env_len)
    cfg["env"]["frames"]      = int(frames)   # tie env frames to sim frames
    cfg["env"]["noise_sigma"] = float(env_noise)
    cfg["env"]["sources"]     = sources_text  # keep as text; builder will parse

    engine = build_engine_from_dict(cfg)

    # live run with callbacks
    st.session_state.stop = False
    last_update = -1

    def cb(t: int):
        nonlocal last_update
        # update UI every chunk frames or at the end
        if (t - last_update) >= chunk or (t == engine.T - 1):
            last_update = t
            # update charts
            draw_energy(energy_ph, engine.hist)
            draw_combined_heatmap(combo_ph, engine.env[:t+1], engine.S[:t+1])
            # log
            log_box.write(f"t={t} / {engine.T-1} | E_cell={engine.hist.E_cell[-1]:.4f} | E_flux={engine.hist.E_flux[-1]:.4f}")
            # let Streamlit breathe
            time.sleep(0.001)
        # allow user stop
        if st.session_state.stop:
            raise RuntimeError("Stopped by user")

    try:
        engine.run(progress_cb=cb)
        # final refresh
        draw_energy(energy_ph, engine.hist)
        draw_combined_heatmap(combo_ph, engine.env, engine.S)
        log_box.write("Done.")
    except RuntimeError as ex:
        if "Stopped by user" in str(ex):
            log_box.write("Stopped.")
        else:
            raise

# kick off
if run_btn:
    run()
else:
    # initial empty renders (optional)
    pass