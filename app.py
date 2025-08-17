# app.py
import json
import time
import numpy as np
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine
from smooth_plot_helpers import (
    ensure_combo_fig,
    update_combo_fig,
    draw_energy_timeseries_live,
)

import inspect
print("ensure_combo_fig ->", inspect.signature(ensure_combo_fig))

st.set_page_config(page_title="Fuka • Free‑Energy Simulation", layout="wide")

# ----------------------
# Sidebar controls
# ----------------------
st.title("Fuka • Free‑Energy Simulation")

cfg = default_config()

with st.sidebar:
    st.header("Run Controls")

    # --- Basics ---
    seed   = st.number_input("Seed", min_value=0, max_value=10_000_000, value=int(cfg["seed"]), step=1)
    frames = st.number_input("Frames", min_value=200, max_value=50_000, value=int(cfg["frames"]), step=200)
    space  = st.number_input("Substrate cells (space)", min_value=16, max_value=1024, value=int(cfg["space"]), step=16)

    st.divider()
    st.subheader("Physics")
    k_flux  = st.slider("k_flux (boundary pump)", 0.0, 1.0, float(cfg["k_flux"]), 0.01)
    k_motor = st.slider("k_motor (motor noise @ boundary)", 0.0, 5.0, float(cfg["k_motor"]), 0.01)
    diffuse = st.slider("diffuse", 0.0, 0.5, float(cfg["diffuse"]), 0.005)
    decay   = st.slider("decay", 0.0, 0.2, float(cfg["decay"]), 0.001)

    st.divider()
    st.subheader("Connections")
    n_conn  = st.number_input("Total connections (n_conn)", 0, 100_000, int(cfg.get("n_conn", 128)), 8)
    n_seed  = st.number_input("Seed per unit (n_seed)",    0, 1024,     int(cfg.get("n_seed", 4)), 1)

    col_len = st.columns(2)
    with col_len[0]:
        ell_min = st.number_input("ell_min", 0.0, 1.0, float(cfg.get("ell_min", 0.05)), 0.005)
    with col_len[1]:
        ell_max = st.number_input("ell_max", 0.0, 1.0, float(cfg.get("ell_max", 0.40)), 0.005)

    col_tau = st.columns(2)
    with col_tau[0]:
        tau_min = st.number_input("tau_min", 1.0, 10_000.0, float(cfg.get("tau_min", 1.0)), 1.0)
    with col_tau[1]:
        tau_max = st.number_input("tau_max", 1.0, 10_000.0, float(cfg.get("tau_max", 16.0)), 1.0)

    col_dyn = st.columns(2)
    with col_dyn[0]:
        conn_grow_every  = st.number_input("grow_every (frames)", 1, 1_000_000, int(cfg.get("conn_grow_every", 50)), 1)
        conn_grow_budget = st.number_input("grow_budget",          0, 10_000,    int(cfg.get("conn_grow_budget", 4)), 1)
    with col_dyn[1]:
        conn_prune_every  = st.number_input("prune_every (frames)", 1, 1_000_000, int(cfg.get("conn_prune_every", 100)), 1)
        conn_prune_thresh = st.number_input("prune_thresh (energy)", -1e9, 1e9, float(cfg.get("conn_prune_thresh", 0.0)), 0.01)

    # Simple guards to keep mins <= maxes
    if ell_min > ell_max: ell_min, ell_max = ell_max, ell_min
    if tau_min > tau_max: tau_min, tau_max = tau_max, tau_min

    st.divider()
    st.subheader("Environment")
    env_len   = st.number_input("Env length (x)", min_value=int(space), max_value=4096, value=int(cfg["env"]["length"]), step=int(space))
    env_noise = st.slider("Env noise σ", 0.0, 0.2, float(cfg["env"]["noise_sigma"]), 0.005)

    st.caption("Sources JSON (e.g. moving peaks). Edit freely.")
    default_sources = json.dumps(cfg["env"]["sources"], indent=2)
    sources_text = st.text_area("env.sources JSON", value=default_sources, height=220)

    # Parse sources JSON safely
    try:
        sources = json.loads(sources_text)
        if not isinstance(sources, list):
            raise ValueError("sources must be a list of dicts")
    except Exception as e:
        st.error(f"Invalid sources JSON: {e}")
        sources = cfg["env"]["sources"]  # fallback

    st.divider()
    chunk = st.slider("Update chunk (frames per UI update)", 10, 500, int(cfg.get("chunk", 150)), 10)
    live  = st.toggle("Live streaming", value=bool(cfg.get("live", True)))

# --- write back into cfg dict so the rest of the app sees updates ---
cfg.update({
    "seed": int(seed),
    "frames": int(frames),
    "space": int(space),
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "diffuse": float(diffuse),
    "decay": float(decay),

    "n_conn": int(n_conn),
    "n_seed": int(n_seed),
    "ell_min": float(ell_min),
    "ell_max": float(ell_max),
    "tau_min": float(tau_min),
    "tau_max": float(tau_max),
    "conn_grow_every": int(conn_grow_every),
    "conn_grow_budget": int(conn_grow_budget),
    "conn_prune_every": int(conn_prune_every),
    "conn_prune_thresh": float(conn_prune_thresh),

    "chunk": int(chunk),
    "live": bool(live),

    "env": {
        "length": int(env_len),
        "frames": int(frames),  # keep env timeline aligned to sim frames
        "noise_sigma": float(env_noise),
        "sources": sources,
    },
})
# Parse sources JSON safely
try:
    sources = json.loads(sources_text)
    if not isinstance(sources, list):
        raise ValueError("sources must be a list of dicts.")
except Exception as e:
    st.error(f"Invalid sources JSON: {e}")
    sources = cfg["env"]["sources"]

# Build run configuration dict
# Build run configuration dict (mirrors the sidebar controls)
user_cfg = {
    "seed": int(seed),
    "frames": int(frames),
    "space": int(space),

    # physics
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "diffuse": float(diffuse),
    "decay": float(decay),

    # connections (new)
    "n_conn": int(n_conn),
    "n_seed": int(n_seed),
    "ell_min": float(ell_min),
    "ell_max": float(ell_max),
    "tau_min": float(tau_min),
    "tau_max": float(tau_max),
    "conn_grow_every": int(conn_grow_every),
    "conn_grow_budget": int(conn_grow_budget),
    "conn_prune_every": int(conn_prune_every),
    "conn_prune_thresh": float(conn_prune_thresh),

    # env
    "env": {
        "length": int(env_len),
        "frames": int(frames),           # keep aligned with sim
        "noise_sigma": float(env_noise),
        "sources": sources,              # already parsed JSON (list of dicts)
    },

    # UI/run loop helpers (optional but handy)
    "chunk": int(chunk),
    "live": bool(live),
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

    with ph_info:
        st.write("**Run complete**")
        st.json(user_cfg)

if run_btn:
    run_live()