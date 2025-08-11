# app.py
# Streamlit UI with live energies and a zoomable Plotly figure
# (stacked heatmaps of E and S with an Overlay toggle), plus kernel views.

from __future__ import annotations
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sim_core import make_engine, default_config, Engine


st.set_page_config(page_title="Fuka – Free‑Energy Gradient Simulation (Live)",
                   layout="wide")


# ------------- small helpers -------------

def clamp(v, lo, hi): return max(lo, min(hi, v))


def sources_from_text(txt: str):
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data, "OK"
    except Exception as e:
        return None, f"JSON error: {e}"
    return None, "JSON must be a list"


def fig_stacked(env: np.ndarray, subs: np.ndarray, title: str = "") -> go.Figure:
    """Two heatmaps with shared x (time) for zooming."""
    T_env, X_env = env.shape
    T_sub, X_sub = subs.shape

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=("Environment E(t,x)", "Substrate S(t,x)"))

    fig.add_trace(go.Heatmap(
        z=env.T, colorscale="Viridis", colorbar=dict(title="E"),
        showscale=True), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=subs.T, colorscale="Viridis", colorbar=dict(title="S"),
        showscale=True), row=2, col=1)

    fig.update_xaxes(title_text="frame", row=2, col=1)
    fig.update_yaxes(title_text="space", row=1, col=1)
    fig.update_yaxes(title_text="space", row=2, col=1)
    if title:
        fig.update_layout(title=title)
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def fig_overlay(env: np.ndarray, subs: np.ndarray, alpha_env=0.9, alpha_sub=0.9) -> go.Figure:
    """One figure with both heatmaps overlayed (two colorbars)."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=env.T, colorscale="Viridis",
                             colorbar=dict(title="E"), opacity=alpha_env))
    fig.add_trace(go.Heatmap(z=subs.T, colorscale="Plasma",
                             colorbar=dict(title="S"), opacity=alpha_sub))
    fig.update_xaxes(title_text="frame")
    fig.update_yaxes(title_text="space")
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10),
                      title="Overlay: E(t,x) + S(t,x)")
    return fig


def fig_kernel_bar(w: np.ndarray) -> go.Figure:
    x = np.arange(len(w)) - (len(w)//2)
    fig = go.Figure(go.Bar(x=x, y=w))
    fig.update_layout(title="Gate kernel (sensing DoF)",
                      xaxis_title="offset (frames, past→right)",
                      yaxis_title="weight", height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def fig_kernel_timeline(times: np.ndarray, W: np.ndarray) -> go.Figure:
    # W: [snapshots, kernel_len]
    fig = go.Figure(go.Heatmap(z=W, colorscale="RdBu",
                               zmid=0.0,
                               colorbar=dict(title="w"),
                               x=np.arange(W.shape[1]) - (W.shape[1]//2),
                               y=times))
    fig.update_layout(title="Kernel timeline (snapshots over run)",
                      xaxis_title="offset", yaxis_title="frame (snapshot)",
                      height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------- sidebar controls -------------

cfg = default_config()

with st.sidebar:
    st.header("Controls")

    seed   = st.number_input("Seed", 0, 10_000, value=int(cfg["seed"]), step=1)
    frames = st.number_input("Frames", 100, 50_000, value=int(cfg["frames"]), step=100)
    space  = st.number_input("Space (cells)", 16, 512, value=int(cfg["space"]), step=1)

    k_flux  = st.number_input("k_flux (boundary flux)", 0.0, 2.0, value=float(cfg["k_flux"]), step=0.01)
    k_motor = st.number_input("k_motor (motor explore)", 0.0, 5.0, value=float(cfg["k_motor"]), step=0.10)
    motor_noise = st.number_input("motor_noise", 0.0, 1.0, value=float(cfg["motor_noise"]), step=0.01)
    c_motor = st.number_input("c_motor (motor work cost)", 0.0, 2.0, value=float(cfg["c_motor"]), step=0.01)

    decay  = st.number_input("decay", 0.0, 0.5, value=float(cfg["decay"]), step=0.001, format="%.3f")
    diffuse= st.number_input("diffuse", 0.0, 1.0, value=float(cfg["diffuse"]), step=0.01)
    band   = st.number_input("band (boundary width)", 1, 8, value=int(cfg["band"]), step=1)

    gate_win = st.number_input("gate_win (half width K)", 1, 64, value=int(cfg["gate_win"]), step=1)
    eta      = st.number_input("eta (kernel LR)", 0.000, 0.5, value=float(cfg["eta"]), step=0.001, format="%.3f")
    ema_beta = st.number_input("ema_beta (baseline EMA)", 0.00, 1.00, value=float(cfg["ema_beta"]), step=0.01)
    lam_l1   = st.number_input("lam_l1 (L1 shrink)", 0.00, 1.00, value=float(cfg["lam_l1"]), step=0.01)
    prune_th = st.number_input("prune_thresh", 0.00, 1.00, value=float(cfg["prune_thresh"]), step=0.01)

    st.subheader("Environment")
    env_len  = st.number_input("Env length", 32, 2048, value=int(cfg["env"]["length"]), step=16)
    env_sig  = st.number_input("Env noise σ", 0.0, 1.0, value=float(cfg["env"]["noise_sigma"]), step=0.01)

    st.caption("Sources JSON (list)")
    sources_txt = st.text_area(
        "Edit env sources",
        value=json.dumps(cfg["env"]["sources"], indent=2),
        height=180
    )
    sources, src_msg = sources_from_text(sources_txt)
    src_ok = sources is not None
    st.button("Sources OK" if src_ok else src_msg, disabled=not src_ok)

    st.subheader("Run")
    chunk = st.slider("Refresh chunk (frames)", 10, 500, 150, 10)
    live  = st.checkbox("Live streaming", True)

# ------------- run button -------------
run = st.button("Run / Rerun", type="primary")

# ------------- holders / placeholders -------------

msg_ph = st.empty()
energy_ph = st.empty()
layout_col1, layout_col2 = st.columns(2)
with layout_col1:
    env_ph = st.empty()
with layout_col2:
    subs_ph = st.empty()

ui_mode = st.radio("Stacked vs Overlay", ["Stacked", "Overlay"], horizontal=True)
ker_bar_ph = st.empty()
ker_tl_ph  = st.empty()


def build_engine() -> Engine:
    cfg_dict = dict(
        seed=seed, frames=int(frames), space=int(space),
        k_flux=float(k_flux), k_motor=float(k_motor), motor_noise=float(motor_noise),
        c_motor=float(c_motor), decay=float(decay), diffuse=float(diffuse), band=int(band),
        gate_win=int(gate_win), eta=float(eta), ema_beta=float(ema_beta),
        lam_l1=float(lam_l1), prune_thresh=float(prune_th),
        env=dict(length=int(env_len), frames=int(frames), noise_sigma=float(env_sig),
                 sources=(sources if src_ok else cfg["env"]["sources"]))
    )
    return make_engine(cfg_dict)


def draw_energies(hist):
    x = np.array(hist.t, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hist.E_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=hist.E_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=hist.E_flux, name="E_flux", mode="lines"))
    fig.update_layout(title="Energies (live)", height=350, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", x=0.02, y=1.10))
    energy_ph.plotly_chart(fig, use_container_width=True)


def draw_fields(env_mat, subs_mat):
    if ui_mode == "Stacked":
        fig = fig_stacked(env_mat, subs_mat, "")
    else:
        fig = fig_overlay(env_mat, subs_mat, 0.9, 0.9)

    # show both in the two slots (for symmetry) when stacked, otherwise show overlay in the right
    if ui_mode == "Stacked":
        env_ph.plotly_chart(fig, use_container_width=True)
        subs_ph.empty()
    else:
        subs_ph.plotly_chart(fig, use_container_width=True)
        env_ph.empty()


def draw_kernel(engine: Engine):
    ker_bar_ph.plotly_chart(fig_kernel_bar(engine.w), use_container_width=True)
    if len(engine.w_hist) > 0:
        times = np.array(engine.w_hist_times, dtype=int)
        W = np.vstack(engine.w_hist)
        ker_tl_ph.plotly_chart(fig_kernel_timeline(times, W), use_container_width=True)


def run_live():
    engine = build_engine()
    msg_ph.success("Starting…")

    # local buffers for live plot
    last_frame_shown = -1

    def cb(t):
        nonlocal last_frame_shown
        if (t - last_frame_shown) >= chunk or (t == engine.T - 1):
            last_frame_shown = t

            # downsample env to current t for speed (or show full to t)
            env_view = engine.env[:t+1]
            subs_view = engine.S[:t+1]

            draw_energies(engine.hist)
            draw_fields(env_view, subs_view)
            draw_kernel(engine)

    engine.run(progress_cb=cb, snapshot_every=max(20, chunk))
    msg_ph.success("Done!")
    # final redraw
    draw_energies(engine.hist)
    draw_fields(engine.env, engine.S)
    draw_kernel(engine)


if run:
    run_live()