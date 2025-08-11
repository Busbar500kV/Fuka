# app.py
# Streamlit UI with live streaming plots and JSON-editable sources.

import importlib, inspect, json, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- force a clean import, then import stable API names ---
import sim_core as sc_mod
importlib.reload(sc_mod)
from sim_core import Engine, make_engine, default_config

# --- show import diagnostics (so we see what got imported) ---
with st.expander("Import diagnostics", expanded=False):
    st.write("sim_core path:", Path(sc_mod.__file__).as_posix())
    st.code("\n".join(sorted([n for n in dir(sc_mod) if not n.startswith("_")])))

st.set_page_config(page_title="Fuka — Free‑Energy Simulation", layout="wide")

st.title("Fuka — Free‑Energy Gradient Demo (live)")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Simulation parameters")

    # Keep a dict config in session_state
    if "cfg" not in st.session_state:
        st.session_state.cfg = default_config()
        # Adjust defaults to match your experiment closely
        st.session_state.cfg.update({
            "frames": 3000,
            "space": 64,
            "k_flux": 0.08,
            "k_motor": 0.40,
            "k_noise": 0.02,
            "decay": 0.01,
            "diffuse": 0.15,
            "band": 3,
            "env": {
                "length": 512,
                "frames": 3000,
                "noise_sigma": 0.0,
                "sources": [
                    {"kind": "constant_patch", "amp": 1.0, "center": 15, "width": 10}
                ],
            },
        })

    cfg = st.session_state.cfg

    # Top-level knobs
    cfg["seed"]   = int(st.number_input("seed", 0, 10_000, int(cfg.get("seed", 0)), 1))
    cfg["frames"] = int(st.number_input("frames", 100, 50_000, int(cfg.get("frames", 3000)), 100))
    cfg["space"]  = int(st.number_input("space (substrate cells)", 8, 1024, int(cfg.get("space", 64)), 1))

    st.markdown("---")
    cfg["k_flux"]  = float(st.number_input("k_flux (boundary flux)", 0.0, 5.0, float(cfg.get("k_flux", 0.08)), 0.01))
    cfg["k_motor"] = float(st.number_input("k_motor (motor explore)", 0.0, 5.0, float(cfg.get("k_motor", 0.40)), 0.01))
    cfg["k_noise"] = float(st.number_input("k_noise (direct band noise)", 0.0, 1.0, float(cfg.get("k_noise", 0.02)), 0.01))
    cfg["decay"]   = float(st.number_input("decay", 0.0, 1.0, float(cfg.get("decay", 0.01)), 0.001))
    cfg["diffuse"] = float(st.number_input("diffuse", 0.0, 1.0, float(cfg.get("diffuse", 0.15)), 0.001))
    cfg["band"]    = int(st.number_input("band (boundary width)", 1, 32, int(cfg.get("band", 3)), 1))

    st.markdown("---")
    st.subheader("Environment")
    env = cfg.setdefault("env", {})
    env["length"]      = int(st.number_input("env.length", 16, 4096, int(env.get("length", 512)), 1))
    env["frames"]      = int(st.number_input("env.frames", 100, 50_000, int(env.get("frames", cfg["frames"])), 100))
    env["noise_sigma"] = float(st.number_input("env.noise_sigma", 0.0, 1.0, float(env.get("noise_sigma", 0.0)), 0.01))

    st.caption("Edit sources JSON below. Example provided in the help box.")
    default_sources_example = [
        {"kind": "constant_patch", "amp": 1.0, "center": 15, "width": 10},
        # {"kind": "moving_peak", "amp": 0.6, "speed": 0.08, "width": 6.0, "start": 340},
        # {"kind": "constant_uniform", "amp": 0.05},
    ]
    raw_sources = st.text_area(
        "env.sources (JSON list)",
        value=json.dumps(env.get("sources", default_sources_example), indent=2),
        height=220,
    )
    try:
        env["sources"] = json.loads(raw_sources)
        sources_ok = True
    except Exception as e:
        sources_ok = False
        st.error(f"Invalid JSON for sources: {e}")

    st.markdown("---")
    live_mode = st.checkbox("Live update while running", value=True)
    chunk     = int(st.number_input("Redraw every N frames", 1, 500, 25, 1))
    go_btn    = st.button("Run simulation", type="primary")


# -------------------------
# Main area placeholders
# -------------------------
col_top = st.columns([2, 1])
with col_top[0]:
    st.subheader("Substrate field S(t, x) — last frame")
    placeholder_img = st.empty()

with col_top[1]:
    st.subheader("Energies")
    placeholder_chart = st.empty()
    placeholder_table = st.empty()

st.subheader("Environment slice vs Substrate (last frame)")
placeholder_env = st.empty()

status_box = st.empty()


# -------------------------
# Helpers for plotting
# -------------------------
def plot_last_frame(S_last: np.ndarray):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(S_last, lw=2)
    ax.set_title("Substrate S at last frame")
    ax.set_xlabel("x")
    ax.set_ylabel("energy")
    st.pyplot(fig)
    plt.close(fig)

def plot_env_vs_subs(env_row: np.ndarray, S_last: np.ndarray):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(env_row, lw=1, label="env (resampled)")
    ax.plot(S_last, lw=2, label="substrate")
    ax.legend()
    ax.set_title("Env row vs Substrate (last frame)")
    st.pyplot(fig)
    plt.close(fig)

def plot_energies(hist):
    df = pd.DataFrame({
        "t": hist.t,
        "E_cell": hist.E_cell,
        "E_env":  hist.E_env,
        "E_flux": hist.E_flux,
    })
    df = df.set_index("t")
    placeholder_chart.line_chart(df)


def redraw(engine: Engine, t: int):
    # last frame data
    S_last = engine.S[t]
    env_row = engine.env[min(t, engine.env.shape[0]-1)]
    # resample env row for overlay plot
    if engine.env.shape[1] != engine.X:
        idx = (np.arange(engine.X) * engine.env.shape[1] // engine.X) % engine.env.shape[1]
        env_row = env_row[idx]

    placeholder_img.empty()
    plot_last_frame(S_last)
    placeholder_env.empty()
    plot_env_vs_subs(env_row, S_last)
    plot_energies(engine.hist)

    # small info table
    data = {
        "t": [t],
        "E_cell": [engine.hist.E_cell[-1]],
        "E_env":  [engine.hist.E_env[-1]],
        "E_flux": [engine.hist.E_flux[-1]],
    }
    placeholder_table.dataframe(pd.DataFrame(data))


# -------------------------
# Run button
# -------------------------
if go_btn and sources_ok:
    # Build engine fresh each run
    engine = make_engine(cfg)

    # Streamed or batch run
    if live_mode:
        last = {"t": -1}
        def cb(t):
            if t == engine.T - 1 or (t - last["t"]) >= chunk:
                last["t"] = t
                status_box.info(f"Running… frame {t+1}/{engine.T}")
                redraw(engine, t)

        engine.run(progress_cb=cb)
        redraw(engine, engine.T - 1)
        status_box.success("Done!")
    else:
        engine.run()
        redraw(engine, engine.T - 1)
        status_box.success("Done!")

else:
    st.info("Adjust parameters on the left, then press **Run simulation**.")