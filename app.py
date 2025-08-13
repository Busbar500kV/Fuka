# app.py
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")

# ----------------------------
# Helpers: figure constructors
# ----------------------------
def make_energy_fig():
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        margin=dict(l=40, r=10, t=30, b=40),
        height=260,
        xaxis_title="t",
        yaxis_title="Energy",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))
    fig.add_scatter(name="E_cell",  x=[], y=[])
    fig.add_scatter(name="E_env",   x=[], y=[])
    fig.add_scatter(name="E_flux",  x=[], y=[])
    return fig

def make_combo_heatmap(X_env, X_space, T):
    # One figure, two heatmaps with opacity controls + shared color scale
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        margin=dict(l=40, r=10, t=30, b=40),
        height=420,
        xaxis_title="x",
        yaxis_title="t",
    ))
    # Substrate (S)
    fig.add_trace(go.Heatmap(
        z=np.zeros((min(1, T), X_space)),
        colorscale="Viridis",
        showscale=True,
        name="Substrate S",
        opacity=0.95
    ))
    # Environment (E, resampled view)
    fig.add_trace(go.Heatmap(
        z=np.zeros((min(1, T), X_space)),
        colorscale="Magma",
        showscale=False,
        name="Env E (resampled)",
        opacity=0.45
    ))
    return fig

def update_energy_fig(fig, hist):
    x = hist.t
    if len(x) == 0:
        return fig
    fig.data[0].x = x; fig.data[0].y = hist.E_cell
    fig.data[1].x = x; fig.data[1].y = hist.E_env
    fig.data[2].x = x; fig.data[2].y = hist.E_flux
    return fig

def update_combo_heatmap(fig, env_rows, S_rows):
    """
    env_rows: (n_rows, X_space)
    S_rows:   (n_rows, X_space)
    We overwrite both heatmaps in the SAME figure (no new figures).
    """
    if env_rows.size == 0 or S_rows.size == 0:
        return fig
    # Normalize per-pane for contrast
    S = S_rows
    E = env_rows
    def norm(a):
        rng = a.max() - a.min()
        return (a - a.min()) / (rng + 1e-9)
    S_n = norm(S)
    E_n = norm(E)
    # Trace 0 = substrate, Trace 1 = env
    fig.data[0].z = S_n
    fig.data[1].z = E_n
    return fig

def make_kernel_fig():
    fig = go.Figure(layout=dict(
        template="plotly_dark",
        margin=dict(l=40, r=10, t=30, b=40),
        height=220,
        xaxis_title="kernel tap",
        yaxis_title="w",
    ))
    fig.add_bar(x=[], y=[], name="gate kernel")
    return fig

def update_kernel_fig(fig, w):
    x = list(range(len(w)))
    fig.data[0].x = x
    fig.data[0].y = w.tolist()
    return fig

# ----------------------------
# Sidebar controls (knobs)
# ----------------------------
st.sidebar.header("Config")

cfg = default_config()   # plain dict

# General
seed   = st.sidebar.number_input("seed", 0, 10_000, int(cfg["seed"]))
frames = st.sidebar.number_input("frames", 100, 50_000, int(cfg["frames"]), step=200)
space  = st.sidebar.number_input("space (substrate length)", 8, 2048, int(cfg["space"]))
band   = st.sidebar.number_input("boundary band (cells)", 1, 32, int(cfg["band"]))
decay  = st.sidebar.number_input("decay", 0.0, 1.0, float(cfg["decay"]), step=0.01, format="%.4f")
diffuse= st.sidebar.number_input("diffuse", 0.0, 1.0, float(cfg["diffuse"]), step=0.01, format="%.4f")
k_flux = st.sidebar.number_input("k_flux (env→cell pump)", 0.0, 5.0, float(cfg["k_flux"]), step=0.01)
k_motor= st.sidebar.number_input("k_motor (random motor kick)", 0.0, 5.0, float(cfg["k_motor"]), step=0.01)
k_noise= st.sidebar.number_input("k_noise (process noise)", 0.0, 1.0, float(cfg["k_noise"]), step=0.01)
cap    = st.sidebar.number_input("cap (clip S)", 0.0, 50.0, float(cfg["cap"]), step=0.5)
bspd   = st.sidebar.number_input("boundary_speed", 0.0, 2.0, float(cfg["boundary_speed"]), step=0.01)

# Kernel (general connection)
st.sidebar.subheader("Gate Kernel")
ker_enabled = st.sidebar.checkbox("enabled", value=bool(cfg["kernel"]["enabled"]))
ker_radius  = st.sidebar.number_input("radius", 0, 32, int(cfg["kernel"]["radius"]))
ker_lr      = st.sidebar.number_input("lr", 0.0, 1.0, float(cfg["kernel"]["lr"]), step=0.0005, format="%.5f")
ker_l2      = st.sidebar.number_input("l2", 0.0, 1.0, float(cfg["kernel"]["l2"]), step=0.0005, format="%.5f")
ker_k       = st.sidebar.number_input("k_gate", 0.0, 5.0, float(cfg["kernel"]["k_gate"]), step=0.01)

# Environment
st.sidebar.subheader("Environment")
env_len   = st.sidebar.number_input("env.length", 8, 4096, int(cfg["env"]["length"]))
env_noise = st.sidebar.number_input("env.noise_sigma", 0.0, 1.0, float(cfg["env"]["noise_sigma"]), step=0.01)
env_base  = st.sidebar.number_input("env.baseline", 0.0, 10.0, float(cfg["env"].get("baseline", 0.0)), step=0.1)
env_scale = st.sidebar.number_input("env.amp_scale", 0.0, 10.0, float(cfg["env"].get("amp_scale", 1.0)), step=0.1)

st.sidebar.caption("Env sources (JSON list). Example:\n"
                   "[{\"kind\":\"moving_peak\",\"amp\":2.0,\"speed\":0.05,\"width\":6.0,\"start\":15}]")
sources_json = st.sidebar.text_area(
    "env.sources JSON",
    value=json.dumps(cfg["env"]["sources"], indent=2),
    height=180
)

# Chunk size
chunk = st.sidebar.number_input("UI update chunk (frames)", 10, 2000, 150, step=10)

# Build config dict back
user_cfg = {
    "seed": seed, "frames": frames, "space": space,
    "band": band, "decay": decay, "diffuse": diffuse,
    "k_flux": k_flux, "k_motor": k_motor, "k_noise": k_noise,
    "cap": cap, "boundary_speed": bspd,
    "kernel": {
        "enabled": ker_enabled,
        "radius": ker_radius,
        "lr": ker_lr,
        "l2": ker_l2,
        "init": 0.0,
        "k_gate": ker_k,
    },
    "env": {
        "length": env_len,
        "frames": frames,
        "noise_sigma": env_noise,
        "baseline": env_base,
        "amp_scale": env_scale,
        "sources": []
    }
}

# Parse source JSON safely
try:
    parsed = json.loads(sources_json)
    if isinstance(parsed, list):
        user_cfg["env"]["sources"] = parsed
except Exception:
    st.sidebar.error("Invalid env.sources JSON — using defaults.")
    user_cfg["env"]["sources"] = default_config()["env"]["sources"]

# ----------------------------
# Main layout
# ----------------------------
col_top  = st.container()
col_mid1 = st.container()
col_mid2 = st.container()

with col_top:
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        st.markdown("### Run")
        go_btn = st.button("Run live", type="primary", use_container_width=True)
        stop_btn = st.button("Stop (resets run)", use_container_width=True)
        status = st.empty()
    with c2:
        st.markdown("### Combined heatmap: Substrate (Viridis) + Env (Magma)")
        combo_ph = st.empty()
    with c3:
        st.markdown("### Energy")
        energy_ph = st.empty()

with col_mid1:
    st.markdown("### Gate Kernel")
    kernel_ph = st.empty()

# ----------------------------
# Run loop (live)
# ----------------------------
def run_live():
    cfg = make_config_from_dict(user_cfg)
    engine = Engine(cfg)

    # Make figures once
    combo_fig  = make_combo_heatmap(engine.env.shape[1], engine.X, engine.T)
    energy_fig = make_energy_fig()
    kernel_fig = make_kernel_fig()

    combo_ph.plotly_chart(combo_fig, use_container_width=True)
    energy_ph.plotly_chart(energy_fig, use_container_width=True)
    kernel_ph.plotly_chart(kernel_fig, use_container_width=True)

    status.info("Running…")
    rows_env = []
    rows_S   = []

    def cb(t: int):
        # Store rows for heatmap
        rows_env.append(engine._row(t))   # resampled env row used in step()
        rows_S.append(engine.S[t].copy())

        # Redraw every chunk or on last frame
        if (t + 1) % int(chunk) == 0 or t == engine.T - 1:
            env_block = np.vstack(rows_env)
            S_block   = np.vstack(rows_S)
            # Update combined heatmap IN PLACE
            update_combo_heatmap(combo_fig, env_block, S_block)
            combo_ph.plotly_chart(combo_fig, use_container_width=True)
            # Update energy
            update_energy_fig(energy_fig, engine.hist)
            energy_ph.plotly_chart(energy_fig, use_container_width=True)
            # Update kernel
            update_kernel_fig(kernel_fig, engine.kernel.w)
            kernel_ph.plotly_chart(kernel_fig, use_container_width=True)
            # Reset block buffers
            rows_env.clear()
            rows_S.clear()

    engine.run(progress_cb=cb)
    status.success("Done!")

if stop_btn:
    st.experimental_rerun()

if go_btn:
    run_live()