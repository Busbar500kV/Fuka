# app.py
import json
import time
import numpy as np
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine
from core.plot import (
    draw_energy_timeseries,
    draw_overlay_last_frame,
    draw_heatmap_full,
)
from core.smooth_plot_helpers import (
    ensure_combo_fig,
    update_combo_fig,
    draw_energy_timeseries_live,
)
st.set_page_config(page_title="Fuka • Free‑Energy Simulation", layout="wide")

# ----------------------
# Sidebar controls
# ----------------------
st.title("Fuka • Free‑Energy Simulation")

cfg = default_config()

with st.sidebar:
    st.header("Run Controls")
    seed   = st.number_input("Seed", min_value=0, max_value=10_000_000, value=cfg["seed"], step=1)
    frames = st.number_input("Frames", min_value=200, max_value=50_000, value=cfg["frames"], step=200)
    space  = st.number_input("Substrate cells (space)", min_value=16, max_value=1024, value=cfg["space"], step=16)

    st.divider()
    st.subheader("Physics")
    k_flux   = st.slider("k_flux (boundary pump)", 0.0, 1.0, float(cfg["k_flux"]), 0.01)
    k_motor  = st.slider("k_motor (motor noise @ boundary)", 0.0, 5.0, float(cfg["k_motor"]), 0.01)
    diffuse  = st.slider("diffuse", 0.0, 0.5, float(cfg["diffuse"]), 0.005)
    decay    = st.slider("decay", 0.0, 0.2, float(cfg["decay"]), 0.001)

    st.divider()
    st.subheader("Environment")
    env_len  = st.number_input("Env length (x)", min_value=space, max_value=4096, value=cfg["env"]["length"], step=space)
    env_noise= st.slider("Env noise σ", 0.0, 0.2, float(cfg["env"]["noise_sigma"]), 0.005)

    st.caption("Sources JSON (e.g. moving peaks). Edit freely.")
    default_sources = json.dumps(cfg["env"]["sources"], indent=2)
    sources_text = st.text_area("env.sources JSON", value=default_sources, height=220)

    st.divider()
    chunk = st.slider("Update chunk (frames per UI update)", 10, 500, 150, 10)
    live  = st.toggle("Live streaming", value=True)

# Parse sources JSON safely
try:
    sources = json.loads(sources_text)
    if not isinstance(sources, list):
        raise ValueError("sources must be a list of dicts.")
except Exception as e:
    st.error(f"Invalid sources JSON: {e}")
    sources = cfg["env"]["sources"]

# Build run configuration dict
user_cfg = {
    "seed": seed,
    "frames": int(frames),
    "space": int(space),
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "diffuse": float(diffuse),
    "decay": float(decay),
    "env": {
        "length": int(env_len),
        "frames": int(frames),
        "noise_sigma": float(env_noise),
        "sources": sources,
    },
}

# ----------------------
# Layout
# ----------------------
ph_combo  = st.empty()
ph_energy = st.empty()
ph_info   = st.container()
st.divider()

run_btn = st.button("Run", type="primary", use_container_width=True)



# ----------------------
# Main runner
# ----------------------
def run_live():
    ecfg = make_config_from_dict(user_cfg)
    engine = Engine(ecfg)
    
    combo_key = "combo_fig"
    fig_combo = ensure_combo_fig(combo_key,
    T=engine.T,
    X_env=engine.env.shape[1],
    X_space=engine.S.shape[1],
    height=520,
    title="Env + Substrate (combined, zoomable)"
)
        
    # Ensure a persistent figure exists
    if combo_key not in st.session_state or st.session_state.get("reset_combo", False):
        # Make an initial figure with a 1-row slice so shapes are valid
        env_init  = engine.env[:1, :]
        subs_init = engine.S[:1, :]
        st.session_state[combo_key] = make_combo_fig(env_init, subs_init)
        st.session_state["reset_combo"] = False
    
    def cb(t: int):
    if t % chunk == 0 or t == engine.T - 1:
        update_combo_fig(fig_combo, engine.env[:t+1], engine.S[:t+1])
        ph_combo.plotly_chart(fig_combo, use_container_width=True, theme=None)

        draw_energy_timeseries_live(
            ph_energy,
            engine.hist.t,
            engine.hist.E_cell,
            engine.hist.E_env,
            engine.hist.E_flux,
        )
        
    engine.run(progress_cb=cb if live else None)

    # Final refresh
    # draw_energy_timeseries(ph_energy, engine.hist.t, engine.hist.E_cell, engine.hist.E_env, engine.hist.E_flux)
    # draw_overlay_last_frame(ph_last, engine.env, engine.S)
    # draw_heatmap_full(ph_combo, engine.env, engine.S, title="Environment E(t,x)")
    # draw_heatmap_full(ph_sub_heat, engine.S,   title="Substrate S(t,x)")

    with ph_info:
        st.write("**Run complete**")
        st.json(user_cfg)

if run_btn:
    run_live()