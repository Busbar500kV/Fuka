# app.py
# Streamlit frontend with live streaming, JSON sources, and boundary offset plot.

import json
from typing import Dict, Any, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sim_core import Engine, default_config, FieldCfg


st.set_page_config(page_title="Fuka – Free‑Energy Gradient Simulation (Live)",
                   layout="wide")

st.title("Fuka – Free‑Energy Gradient Simulation (Live)")

# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("Controls")

    # Load defaults (dict) and then expose widgets
    cfg: Dict[str, Any] = default_config()

    seed   = st.number_input("Seed", 0, 10_000, value=cfg["seed"], step=1)
    frames = st.number_input("Frames", 100, 100_000, value=cfg["frames"], step=500)
    space  = st.number_input("Space (cells)", 8, 1024, value=cfg["space"], step=8)

    k_flux   = st.number_input("k_flux", 0.0, 10.0, value=float(cfg["k_flux"]), step=0.01)
    k_motor  = st.number_input("k_motor (offset motor gain)", 0.0, 10.0, value=float(cfg["k_motor"]), step=0.05)
    k_noise  = st.number_input("k_noise (direct band noise)", 0.0, 1.0, value=float(cfg["k_noise"]), step=0.01)
    diffuse  = st.number_input("diffuse", 0.0, 1.0, value=float(cfg["diffuse"]), step=0.01)
    decay    = st.number_input("decay", 0.0, 1.0, value=float(cfg["decay"]), step=0.005)
    band     = st.number_input("boundary band size", 1, 16, value=int(cfg["band"]), step=1)

    st.markdown("### Environment")
    env_len  = st.number_input("Env length", 16, 8192, value=int(cfg["env"]["length"]), step=16)
    env_sig  = st.number_input("Env noise σ", 0.0, 1.0, value=float(cfg["env"]["noise_sigma"]), step=0.01)

    st.markdown("**Sources JSON (list)**")
    default_sources = json.dumps(cfg["env"]["sources"], indent=2)
    sources_txt = st.text_area("Edit env sources", value=default_sources, height=180)
    sources_ok = True
    try:
        sources_list: List[Dict[str, Any]] = json.loads(sources_txt)
        if not isinstance(sources_list, list):
            raise ValueError("Sources JSON must be a list")
        st.success("Sources OK")
    except Exception as e:
        sources_ok = False
        st.error(f"Sources JSON error: {e}")

    live = st.checkbox("Live streaming", value=True)
    chunk = st.slider("Refresh chunk (frames)", 50, 1000, 200, 10)

# Stitch config
cfg_out: Dict[str, Any] = {
    "seed": int(seed),
    "frames": int(frames),
    "space": int(space),
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "k_noise": float(k_noise),
    "diffuse": float(diffuse),
    "decay": float(decay),
    "band": int(band),
    "env": {
        "length": int(env_len),
        "frames": int(frames),          # keep env timeline aligned with sim
        "noise_sigma": float(env_sig),
        "sources": sources_list if sources_ok else FieldCfg().sources,
    },
}

# --------------------------
# Run button
# --------------------------
start = st.button("Start / Restart")

# Placeholders for plots
status_box = st.empty()
chart_energy = st.empty()
col_env, col_subs = st.columns(2)
env_img_pl = col_env.empty()
subs_img_pl = col_subs.empty()
offset_chart = st.empty()

def draw_energy(hist_t, E_cell, E_env, E_flux):
    fig, ax = plt.subplots(figsize=(6.8, 3))
    ax.plot(hist_t, E_cell, label="E_cell")
    ax.plot(hist_t, E_env,  label="E_env")
    ax.plot(hist_t, E_flux, label="E_flux")
    ax.set_xlabel("frame")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    chart_energy.pyplot(fig)
    plt.close(fig)

def draw_heatmap_env(env, upto):
    # env: (T, X_env)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.imshow(env[:upto].T, aspect="auto", origin="lower",
              interpolation="nearest")
    ax.set_title("Environment E(t, x)")
    ax.set_xlabel("frame"); ax.set_ylabel("space")
    env_img_pl.pyplot(fig)
    plt.close(fig)

def draw_heatmap_subs(S, upto):
    # S: (T, X)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.imshow(S[:upto].T, aspect="auto", origin="lower",
              interpolation="nearest")
    ax.set_title("Substrate S(t, x)")
    ax.set_xlabel("frame"); ax.set_ylabel("space")
    subs_img_pl.pyplot(fig)
    plt.close(fig)

def draw_offset(hist_t, o):
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    ax.plot(hist_t, o, lw=1.2)
    ax.set_xlabel("frame"); ax.set_ylabel("offset (env idx)")
    ax.set_title("Boundary offset o(t)")
    ax.grid(True, alpha=0.3)
    offset_chart.pyplot(fig)
    plt.close(fig)

def run_streaming(cfg_dict: Dict[str, Any]):
    status_box.info("Starting…")
    engine = Engine(
        # dataclass wants native types; we pass the plain dict to Engine via Config inside Engine?
        # Here we call Engine with a constructed dict by reusing sim_core.Config via default_config path,
        # but Engine’s __init__ accepts Config, not dict—so we use run_sim pattern:
        # To keep it simple, import Config? Avoid; we can mimic by calling Engine through a tiny wrapper.
        # Easiest: re-create Engine exactly as run_sim does, but inline:
        __import__("sim_core").Config(
            seed=cfg_dict["seed"],
            frames=cfg_dict["frames"],
            space=cfg_dict["space"],
            k_flux=cfg_dict["k_flux"],
            k_motor=cfg_dict["k_motor"],
            k_noise=cfg_dict["k_noise"],
            decay=cfg_dict["decay"],
            diffuse=cfg_dict["diffuse"],
            band=cfg_dict["band"],
            env=FieldCfg(
                length=cfg_dict["env"]["length"],
                frames=cfg_dict["frames"],
                noise_sigma=cfg_dict["env"]["noise_sigma"],
                sources=cfg_dict["env"]["sources"],
            ),
        )
    )  # type: ignore

    # Rebuild engine with the Config constructed above
    from sim_core import Engine as _Engine  # local alias
    engine = _Engine(engine)  # yes: pass the Config instance we just created

    T = cfg_dict["frames"]
    last = [0]

    def progress(t: int):
        if (t - last[0] >= chunk) or (t == T - 1):
            last[0] = t
            status_box.success("Running…")
            # draw
            draw_energy(engine.hist.t, engine.hist.E_cell, engine.hist.E_env, engine.hist.E_flux)
            draw_heatmap_env(engine.env, t + 1)
            draw_heatmap_subs(engine.S,   t + 1)
            draw_offset(engine.hist.t, engine.hist.o)

    engine.run(progress_cb=progress)
    status_box.success("Done!")

# Kick off
if start:
    run_streaming(cfg_out)