# app.py
# Live heatmaps for Environment and Substrate + live energy chart.
# Uses placeholders so figures update in place (no plot stacking).

import importlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# clean import then import stable API
import sim_core as sc_mod
importlib.reload(sc_mod)
from sim_core import make_engine, default_config

st.set_page_config(page_title="Fuka — Free‑Energy Simulation (Live)", layout="wide")
st.title("Fuka — Free‑Energy Gradient Simulation (Live)")

with st.expander("Import diagnostics", expanded=False):
    st.write("sim_core path:", Path(sc_mod.__file__).as_posix())
    st.code("\n".join(sorted([n for n in dir(sc_mod) if not n.startswith('_')])))

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Controls")

    if "cfg" not in st.session_state:
        st.session_state.cfg = default_config()
        # defaults close to your experiment
        st.session_state.cfg.update({
            "frames": 5000,
            "space": 64,
            "k_flux": 0.08,
            "k_motor": 0.40,
            "k_noise": 0.02,
            "decay": 0.01,
            "diffuse": 0.15,
            "band": 3,
            "env": {
                "length": 512,
                "frames": 5000,
                "noise_sigma": 0.0,
                "sources": [
                    {"kind": "constant_patch", "amp": 1.0, "center": 15, "width": 10}
                ],
            },
        })

    cfg = st.session_state.cfg
    cfg["seed"]   = int(st.number_input("Seed", 0, 10_000, int(cfg.get("seed", 0)), 1))
    cfg["frames"] = int(st.number_input("Frames", 100, 50_000, int(cfg.get("frames", 5000)), 100))
    cfg["space"]  = int(st.number_input("Space (cells)", 8, 1024, int(cfg.get("space", 64)), 1))

    st.markdown("---")
    cfg["k_flux"]  = float(st.number_input("k_flux (boundary flux)", 0.0, 5.0, float(cfg.get("k_flux", 0.08)), 0.01))
    cfg["k_motor"] = float(st.number_input("k_motor (motor explore)", 0.0, 5.0, float(cfg.get("k_motor", 0.40)), 0.01))
    cfg["k_noise"] = float(st.number_input("k_noise (band noise)", 0.0, 1.0, float(cfg.get("k_noise", 0.02)), 0.01))
    cfg["decay"]   = float(st.number_input("decay", 0.0, 1.0, float(cfg.get("decay", 0.01)), 0.001))
    cfg["diffuse"] = float(st.number_input("diffuse", 0.0, 1.0, float(cfg.get("diffuse", 0.15)), 0.001))
    cfg["band"]    = int(st.number_input("band (boundary width)", 1, 32, int(cfg.get("band", 3)), 1))

    st.markdown("---")
    st.subheader("Environment")
    env = cfg.setdefault("env", {})
    env["length"]      = int(st.number_input("Env length", 16, 4096, int(env.get("length", 512)), 1))
    env["frames"]      = int(st.number_input("Env frames", 100, 50_000, int(env.get("frames", cfg["frames"])), 100))
    env["noise_sigma"] = float(st.number_input("Env noise σ", 0.0, 1.0, float(env.get("noise_sigma", 0.0)), 0.01))

    st.caption("Edit env sources JSON:")
    sources_default = [
        {"kind": "constant_patch", "amp": 1.0, "center": 15, "width": 10},
        # {"kind": "moving_peak", "amp": 0.6, "speed": 0.08, "width": 6.0, "start": 340},
        # {"kind": "constant_uniform", "amp": 0.05},
    ]
    raw_sources = st.text_area("env.sources (JSON list)",
                               value=json.dumps(env.get("sources", sources_default), indent=2),
                               height=220)
    try:
        env["sources"] = json.loads(raw_sources)
        sources_ok = True
    except Exception as e:
        sources_ok = False
        st.error(f"Invalid JSON: {e}")

    st.markdown("---")
    live_mode = st.checkbox("Live streaming", value=True)
    chunk     = int(st.number_input("Refresh chunk (frames)", 1, 500, 25, 1))
    go        = st.button("Run simulation", type="primary")

# -------------------------
# Placeholders
# -------------------------
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Environment E(t,x)  •  Substrate S(t,x)")
    ph_env_heat = st.empty()
    ph_sub_heat = st.empty()
with col2:
    st.subheader("Energies (live)")
    ph_energy = st.empty()
    ph_status = st.empty()

# -------------------------
# Matplotlib helpers
# -------------------------
def render_env_heatmap(E: np.ndarray):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    im = ax.imshow(E.T, aspect="auto", origin="lower")
    ax.set_xlabel("frame"); ax.set_ylabel("space")
    ax.set_title("Environment E(t,x)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def render_substrate_heatmap(S_part: np.ndarray):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    im = ax.imshow(S_part.T, aspect="auto", origin="lower")
    ax.set_xlabel("frame"); ax.set_ylabel("space")
    ax.set_title("Substrate S(t,x)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def render_energy(hist):
    df = pd.DataFrame({
        "t": hist.t,
        "E_cell": hist.E_cell,
        "E_env":  hist.E_env,
        "E_flux": hist.E_flux,
    }).set_index("t")
    ph_energy.line_chart(df)

# -------------------------
# Run
# -------------------------
if go and sources_ok:
    engine = make_engine(cfg)

    # Environment heatmap is static → draw once
    ph_env_heat.pyplot(render_env_heatmap(engine.env), clear_figure=True)

    if live_mode:
        last = {"t": -1}
        def cb(t):
            if t == engine.T - 1 or (t - last["t"]) >= chunk:
                last["t"] = t
                # update substrate heatmap with data up to t
                ph_sub_heat.pyplot(render_substrate_heatmap(engine.S[:t+1]), clear_figure=True)
                render_energy(engine.hist)
                ph_status.info(f"Running… frame {t+1}/{engine.T}")

        engine.run(progress_cb=cb)
        # final refresh
        ph_sub_heat.pyplot(render_substrate_heatmap(engine.S), clear_figure=True)
        render_energy(engine.hist)
        ph_status.success("Done!")
    else:
        engine.run()
        ph_sub_heat.pyplot(render_substrate_heatmap(engine.S), clear_figure=True)
        render_energy(engine.hist)
        ph_status.success("Done!")
else:
    st.info("Set parameters and press **Run simulation**.")