# app.py — Live Streamlit dashboard for the sliding-gate model (Model A)

import json
import io
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# ensure latest sim_core is used (Streamlit can cache modules)
from importlib import reload
import sim_core as sc
sc = reload(sc)

from sim_core import make_engine, default_config, Engine

st.set_page_config(page_title="Fuka – Free‑Energy Gradient Simulation (Live)", layout="wide")
st.title("Fuka – Free‑Energy Gradient Simulation (Live)")

# ---------- helpers ----------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def draw_energy(history, placeholder):
    t = np.array(history.t, dtype=float)
    if len(t) == 0: 
        placeholder.empty()
        return

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t, history.E_cell, label="E_cell")
    ax.plot(t, history.E_env,  label="E_env")
    ax.plot(t, history.E_flux, label="E_flux")
    ax.set_title("Energies (live)")
    ax.set_xlabel("frame"); ax.set_ylabel("level")
    ax.legend(loc="upper left")
    fig.tight_layout()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)

def draw_heatmap(mat, title, placeholder, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(mat.T, origin="lower", aspect="auto", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("frame"); ax.set_ylabel("space")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)

def draw_kernel(w, placeholder):
    fig, ax = plt.subplots(figsize=(4,2.2))
    ax.bar(np.arange(len(w)), w)
    ax.set_title("Gate kernel (sensing DoF)")
    ax.set_xlabel("offset"); ax.set_ylabel("weight")
    fig.tight_layout()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ---------- sidebar controls ----------
with st.sidebar:
    st.header("Controls")

    cfg: Dict = st.session_state.get("cfg", default_config())

    # core sizes
    cfg["seed"]   = int(st.number_input("Seed", 0, 10_000, int(cfg.get("seed", 0)), 1))
    cfg["frames"] = int(st.number_input("Frames", 100, 50_000, int(cfg.get("frames", 5000)), 100))
    cfg["space"]  = int(st.number_input("Space (cells)", 16, 512, int(cfg.get("space", 64)), 1))

    # physics
    cfg["k_flux"]      = float(st.number_input("k_flux (boundary flux)", 0.0, 5.0, float(cfg.get("k_flux", 0.08)), 0.01))
    cfg["k_motor"]     = float(st.number_input("k_motor (motor explore)", 0.0, 10.0, float(cfg.get("k_motor", 0.40)), 0.01))
    cfg["motor_noise"] = float(st.number_input("motor_noise", 0.0, 1.0, float(cfg.get("motor_noise", 0.02)), 0.01))
    cfg["c_motor"]     = float(st.number_input("c_motor (motor work cost)", 0.0, 1.0, float(cfg.get("c_motor", 0.02)), 0.001))
    cfg["decay"]       = float(st.number_input("decay", 0.0, 0.5, float(cfg.get("decay", 0.01)), 0.001))
    cfg["diffuse"]     = float(st.number_input("diffuse", 0.0, 1.0, float(cfg.get("diffuse", 0.15)), 0.01))
    cfg["band"]        = int(st.number_input("band (boundary width)", 1, 16, int(cfg.get("band", 3)), 1))

    # kernel learning
    cfg["gate_win"]    = int(st.number_input("gate_win (half width K)", 1, 32, int(cfg.get("gate_win", 8)), 1))
    cfg["eta"]         = float(st.number_input("eta (kernel LR)", 0.0, 1.0, float(cfg.get("eta", 0.02)), 0.001))
    cfg["ema_beta"]    = float(st.number_input("ema_beta (baseline EMA)", 0.0, 1.0, float(cfg.get("ema_beta", 0.10)), 0.01))
    cfg["lam_l1"]      = float(st.number_input("lam_l1 (L1 shrink)", 0.0, 0.1, float(cfg.get("lam_l1", 0.001)), 0.001))
    cfg["prune_thresh"]= float(st.number_input("prune_thresh", 0.0, 0.1, float(cfg.get("prune_thresh", 1e-3)), 0.001))

    # env block
    if "env" not in cfg: cfg["env"] = {}
    cfg["env"]["length"]     = int(st.number_input("Env length", 32, 4096, int(cfg["env"].get("length", 512)), 1))
    cfg["env"]["noise_sigma"]= float(st.number_input("Env noise σ", 0.0, 1.0, float(cfg["env"].get("noise_sigma", 0.01)), 0.01))
    cfg["env"]["frames"]     = int(cfg["frames"])

    st.caption("Sources JSON (list)")
    src_default = cfg["env"].get("sources", [
        {"kind":"moving_peak","amp":1.0,"speed":0.0,"width":4.0,"start":340}
    ])
    src_text = st.text_area("Edit env sources", json.dumps(src_default, indent=2), height=160)
    ok = True
    try:
        cfg["env"]["sources"] = json.loads(src_text)
        st.success("Sources OK")
    except Exception as e:
        ok = False
        st.error(f"Invalid JSON: {e}")

    st.session_state["cfg"] = cfg

    live = st.checkbox("Live streaming", value=True)
    chunk = int(st.slider("Refresh chunk (frames)", 10, 500, 100, 10))
    st.markdown("---")
    run_btn = st.button("Run / Rerun", use_container_width=True, disabled=not ok)

# ---------- placeholders ----------
msg_box  = st.empty()
energies = st.empty()
col1, col2 = st.columns(2, gap="large")
env_ph   = col1.empty()
subs_ph  = col2.empty()
ker_ph   = st.empty()

# ---------- run ----------
if run_btn and ok:
    msg_box.info("Starting…")
    engine: Engine = make_engine(cfg)

    # static env heatmap (known ahead)
    msg_box.info("Building environment…")
    draw_heatmap(engine.env, "Environment E(t,x)", env_ph)

    # streaming loop
    msg_box.info("Running…")
    T = engine.T
    last_draw = -1
    vmax_s = None  # autoscale after a few frames

    for t in range(T):
        engine.step(t)

        # draw periodically
        if (t - last_draw) >= chunk or t == T-1:
            last_draw = t
            draw_energy(engine.hist, energies)

            # substrates: only up to current t
            S_part = engine.S[:t+1]
            # fix color scale after some warm-up so it doesn't flicker
            if vmax_s is None and t > max(20, chunk):
                vmax_s = float(np.percentile(S_part, 99))
            draw_heatmap(S_part, "Substrate S(t,x)", subs_ph, vmin=0.0, vmax=vmax_s)
            
            # draw kernel if present
            if hasattr(engine, "w") and engine.w is not None:
                draw_kernel(engine.w, ker_ph)
            else:
                ker_ph.info("Kernel not available yet.")

            # draw_kernel(engine.w, ker_ph)

    msg_box.success("Done!")