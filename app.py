# app.py — Streamlit UI / orchestration only
import json
from typing import Dict, Tuple, Callable

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sim_core import Engine, default_config  # unchanged physics/knobs

# plotting helpers
from plots_full import (
    make_combined_overlay_figure,
    update_combined_overlay_figure,
    make_energy_figure,
    update_energy_figure,
)

st.set_page_config(page_title="Fuka – Free‑Energy Gradient (Live)", layout="wide")


def merge_cfg(base: Dict, patch: Dict) -> Dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def sidebar_controls() -> Dict:
    """Render controls and return a plain dict config."""
    st.sidebar.header("Controls")

    cfg = default_config()

    # Basics
    cfg["seed"]   = st.sidebar.number_input("Seed",   0, 10_000, value=int(cfg["seed"]))
    cfg["frames"] = st.sidebar.number_input("Frames", 100, 50_000, value=int(cfg["frames"]), step=100)
    cfg["space"]  = st.sidebar.number_input("Space (cells)", 16, 512, value=int(cfg["space"]))

    # Physics knobs (exact same names as in sim_core.Config)
    cfg["k_flux"]      = st.sidebar.number_input("k_flux (boundary flux)",    0.0, 2.0, value=float(cfg["k_flux"]), step=0.01)
    cfg["k_motor"]     = st.sidebar.number_input("k_motor (motor explore)",   0.0, 10.0, value=float(cfg["k_motor"]), step=0.01)
    cfg["motor_noise"] = st.sidebar.number_input("motor_noise",               0.0, 1.0, value=float(cfg.get("motor_noise", 0.02)), step=0.01)
    cfg["c_motor"]     = st.sidebar.number_input("c_motor (motor work cost)", 0.0, 5.0, value=float(cfg.get("c_motor", 0.8)), step=0.01)
    cfg["decay"]       = st.sidebar.number_input("decay",                     0.0, 0.5, value=float(cfg["decay"]), step=0.005)
    cfg["diffuse"]     = st.sidebar.number_input("diffuse",                   0.0, 1.0, value=float(cfg["diffuse"]), step=0.01)
    cfg["band"]        = st.sidebar.number_input("band (boundary width)",     1,  16,  value=int(cfg.get("band", 3)))

    # Gate/kernel (kept; harmless if sim_core ignores some)
    cfg["gate_win"]  = st.sidebar.number_input("gate_win (half width K)", 1, 64, value=int(cfg.get("gate_win", 30)))
    cfg["eta"]       = st.sidebar.number_input("eta (kernel LR)",         0.000, 0.5, value=float(cfg.get("eta", 0.02)), step=0.001, format="%.3f")
    cfg["ema_beta"]  = st.sidebar.number_input("ema_beta (baseline EMA)", 0.000, 1.0, value=float(cfg.get("ema_beta", 0.10)), step=0.01)
    cfg["lam_l1"]    = st.sidebar.number_input("lam_l1 (L1 shrink)",      0.00,  1.0, value=float(cfg.get("lam_l1", 0.10)), step=0.01)
    cfg["prune_thresh"] = st.sidebar.number_input("prune_thresh",          0.00,  1.0, value=float(cfg.get("prune_thresh", 0.10)), step=0.01)

    # Environment
    env = cfg.get("env", {})
    env_length = st.sidebar.number_input("Env length", 32, 2048, value=int(env.get("length", 512)))
    env_noise  = st.sidebar.number_input("Env noise σ", 0.0, 1.0, value=float(env.get("noise_sigma", 0.01)), step=0.01)

    sources_json_default = json.dumps(env.get("sources", [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0,  "width": 4.0, "start": 340}
    ]), indent=2)

    st.sidebar.caption("Sources JSON (list)")
    txt = st.sidebar.text_area("Edit env sources", value=sources_json_default, height=240)
    try:
        sources = json.loads(txt)
        st.sidebar.success("Sources OK")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
        sources = env.get("sources", [])

    cfg["env"] = merge_cfg(env, {"length": env_length, "noise_sigma": env_noise, "sources": sources})

    # Live streaming chunk
    chunk = st.sidebar.slider("Refresh chunk (frames)", 20, 400, 150, step=10)

    # Run button
    run = st.sidebar.button("Run / Rerun", use_container_width=True)

    st.session_state["chunk"] = int(chunk)
    return cfg, run


def resample_to_env_width(S: np.ndarray, env_width: int) -> np.ndarray:
    """Resize substrate (T, Xs) to env width (Xe) by linear interpolation along x."""
    T, Xs = S.shape
    if Xs == env_width:
        return S
    x_src = np.linspace(0, 1, Xs)
    x_dst = np.linspace(0, 1, env_width)
    out = np.empty((T, env_width), dtype=S.dtype)
    for t in range(T):
        out[t] = np.interp(x_dst, x_src, S[t])
    return out


def run():
    cfg, pressed = sidebar_controls()

    # Placeholders (single instances reused every update)
    st.title("Fuka – Free‑Energy Gradient Simulation (Live)")
    combo_ph  = st.empty()
    energy_ph = st.empty()
    status    = st.empty()

    if not pressed:
        st.stop()

    # Build engine AFTER user presses Run
    engine = Engine.from_dict(cfg) if hasattr(Engine, "from_dict") else Engine(cfg)  # keep compatibility

    # Create figures once we know sizes
    # Use full env length on x, full time on y for the Heatmap
    Xe = int(engine.env.shape[1])  # env width
    T  = int(engine.env.shape[0])  # timeline
    # Initial data for plots (zeros with correct shape)
    E0 = np.zeros_like(engine.env)
    S0 = np.zeros((T, Xe))

    combo_fig  = make_combined_overlay_figure(E0, S0, title="E(t,x) • S(t,x)")
    energy_fig = make_energy_figure()

    combo_ph.plotly_chart(combo_fig, use_container_width=True)
    energy_ph.plotly_chart(energy_fig, use_container_width=True)
    status.info("Starting…")

    chunk = int(st.session_state.get("chunk", 150))

    last_draw_t = -1

    def on_progress(t: int):
        nonlocal last_draw_t, combo_fig, energy_fig
        if (t - last_draw_t) >= chunk or t == (engine.T - 1):
            last_draw_t = t

            # Current arrays
            E = engine.env[:t+1]                     # (t+1, Xe_env)
            S = engine.S[:t+1]                       # (t+1, Xs)
            S_env = resample_to_env_width(S, E.shape[1])

            update_combined_overlay_figure(combo_fig, E, S_env)
            combo_ph.plotly_chart(combo_fig, use_container_width=True)

            # Energies
            update_energy_figure(
                energy_fig,
                np.array(engine.hist.t),
                np.array(engine.hist.E_cell),
                np.array(engine.hist.E_env),
                np.array(engine.hist.E_flux),
            )
            energy_ph.plotly_chart(energy_fig, use_container_width=True)

            status.success(f"t = {t+1}/{engine.T}")

    # Run the simulation (streaming)
    engine.run(progress_cb=on_progress)

    status.success("Done!")


if __name__ == "__main__":
    run()