# app.py
# Streamlit UI with live combined heatmap and live energy & kernel plots.

import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sim_core import Engine, default_config, make_config_from_dict

st.set_page_config(page_title="Fuka: Free‑Energy Simulation", layout="wide")


# ============= UI helpers =============

def to_sources_json(sources_obj) -> str:
    try:
        return json.dumps(sources_obj, indent=2)
    except Exception:
        return "[]"


def parse_sources(text: str):
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []


def make_dark(fig, title=None, height=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    if title:
        fig.update_layout(title=title)
    if height:
        fig.update_layout(height=height)
    return fig


def draw_energy(ph, hist):
    t = np.array(hist.t, dtype=int)
    e_cell = np.array(hist.E_cell, dtype=float)
    e_env  = np.array(hist.E_env, dtype=float)
    e_flux = np.array(hist.E_flux, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux", mode="lines"))
    fig.update_xaxes(title="frame")
    fig.update_yaxes(title="energy")
    make_dark(fig, "Energies (live)", height=260)
    ph.plotly_chart(fig, use_container_width=True)


def draw_kernel(w, ph):
    if w is None or len(w) == 0:
        ph.info("Kernel disabled.")
        return
    z = np.array([w], dtype=float)
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale="Viridis", showscale=True, zmin=-1.0, zmax=1.0,
        colorbar=dict(title="w")
    ))
    fig.update_xaxes(title="kernel index")
    fig.update_yaxes(showticklabels=False)
    make_dark(fig, "Gate kernel (live)")
    ph.plotly_chart(fig, use_container_width=True)


def draw_combined_heatmap(ph, env_full, S_full, t_now, window):
    T_full = env_full.shape[0]
    x = np.arange(S_full.shape[1], dtype=int)
    t0 = max(0, int(t_now) - int(window) + 1)
    E = env_full[t0:int(t_now)+1, :]
    S = S_full[t0:int(t_now)+1, :]

    def norm01(A):
        return (A - np.nanmin(A)) / (np.nanmax(A) - np.nanmin(A) + 1e-12)

    X_sub = S.shape[1]
    X_env = E.shape[1]
    if X_env != X_sub:
        idx = (np.arange(X_sub) * X_env // X_sub) % X_env
        E = E[:, idx]

    En = norm01(E)
    Sn = norm01(S)
    y = np.arange(t0, int(t_now)+1, dtype=int)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=En, x=x, y=y, name="Env",
        colorscale="Blues", showscale=False, zmin=0, zmax=1, opacity=1.0
    ))
    fig.add_trace(go.Heatmap(
        z=Sn, x=x, y=y, name="Substrate",
        colorscale="Oranges", showscale=False, zmin=0, zmax=1, opacity=0.55
    ))
    fig.update_xaxes(title="space (cells)")
    fig.update_yaxes(title="frame", autorange="reversed")
    make_dark(fig, "Combined Env + Substrate (live, zoomable)")
    ph.plotly_chart(fig, use_container_width=True)


# ============= Sidebar =============

if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()

with st.sidebar:
    st.header("Run")
    seed    = st.number_input("seed", 0, 10_000, value=int(st.session_state.cfg.get("seed", 0)), step=1)
    frames  = st.number_input("frames", 200, 50_000, value=int(st.session_state.cfg.get("frames", 5000)), step=200)
    space   = st.number_input("space (substrate width)", 8, 1024, value=int(st.session_state.cfg.get("space", 64)), step=1)
    chunk   = st.number_input("UI update chunk (frames)", 10, 2000, value=150, step=10)

    st.divider()
    st.subheader("Environment")
    env_len  = st.number_input("env.length", 8, 4096, value=int(st.session_state.cfg.get("env", {}).get("length", 512)), step=1)
    env_noise= st.number_input("env.noise_sigma", 0.0, 1.0, value=float(st.session_state.cfg.get("env", {}).get("noise_sigma", 0.0)), step=0.01, format="%.2f")
    env_base = st.number_input("env.baseline", -5.0, 5.0, value=float(st.session_state.cfg.get("env", {}).get("baseline", 0.0)), step=0.1, format="%.2f")
    env_scale= st.number_input("env.amp_scale", 0.0, 10.0, value=float(st.session_state.cfg.get("env", {}).get("amp_scale", 1.0)), step=0.1, format="%.1f")
    st.caption("Sources JSON (list). Example:\n"
               '[{"kind":"moving_peak","amp":1.0,"speed":0.0,"width":5.0,"start":15}]')
    src_text = st.text_area(
        "env.sources (JSON)",
        value=to_sources_json(st.session_state.cfg.get("env", {}).get("sources", [
            {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 5.0, "start": 15}
        ])),
        height=140
    )

    st.divider()
    st.subheader("Boundary & Motors")
    k_flux   = st.number_input("k_flux (pump)", 0.0, 10.0, value=float(st.session_state.cfg.get("k_flux", 0.08)), step=0.01)
    k_motor  = st.number_input("k_motor (motor drive)", 0.0, 10.0, value=float(st.session_state.cfg.get("k_motor", 0.50)), step=0.01)
    band     = st.number_input("band (cells)", 1, 32, value=int(st.session_state.cfg.get("band", 3)), step=1)
    b_speed  = st.number_input("boundary_speed", 0.0, 2.0, value=float(st.session_state.cfg.get("boundary_speed", 0.04)), step=0.01)

    st.divider()
    st.subheader("Substrate Physics")
    decay    = st.number_input("decay", 0.0, 0.5, value=float(st.session_state.cfg.get("decay", 0.01)), step=0.001, format="%.3f")
    diffuse  = st.number_input("diffuse", 0.0, 1.0, value=float(st.session_state.cfg.get("diffuse", 0.12)), step=0.01)
    k_noise  = st.number_input("k_noise", 0.0, 2.0, value=float(st.session_state.cfg.get("k_noise", 0.00)), step=0.01)
    cap      = st.number_input("cap (saturation)", 0.0, 100.0, value=float(st.session_state.cfg.get("cap", 5.0)), step=0.5)

    st.divider()
    st.subheader("Kernel (general connections)")
    k_enabled= st.checkbox("kernel.enabled", value=bool(st.session_state.cfg.get("kernel", {}).get("enabled", True)))
    k_radius = st.number_input("kernel.radius", 0, 32, value=int(st.session_state.cfg.get("kernel", {}).get("radius", 3)), step=1)
    k_lr     = st.number_input("kernel.lr", 0.0, 1.0, value=float(st.session_state.cfg.get("kernel", {}).get("lr", 1e-3)), step=1e-3, format="%.3f")
    k_l2     = st.number_input("kernel.l2", 0.0, 1.0, value=float(st.session_state.cfg.get("kernel", {}).get("l2", 1e-4)), step=1e-4, format="%.4f")
    k_init   = st.number_input("kernel.init", 0.0, 1.0, value=float(st.session_state.cfg.get("kernel", {}).get("init", 0.0)), step=0.01)
    k_gate   = st.number_input("kernel.k_gate", 0.0, 5.0, value=float(st.session_state.cfg.get("kernel", {}).get("k_gate", 0.15)), step=0.01)

    st.divider()
    view_window = st.number_input("Heatmap view window (frames)", 50, 2000, value=200, step=50)

run_btn = st.button("Run / Restart", use_container_width=True)


# ============= Main area =============

col1, col2 = st.columns([2, 1], gap="large")
with col1:
    combo_ph = st.empty()
    energy_ph = st.empty()
with col2:
    kernel_ph = st.empty()
    diag = st.container()


def build_user_cfg_dict():
    return dict(
        seed=seed,
        frames=frames,
        space=space,
        k_flux=k_flux,
        k_motor=k_motor,
        band=band,
        decay=decay,
        diffuse=diffuse,
        k_noise=k_noise,
        cap=cap,
        boundary_speed=b_speed,
        env=dict(
            length=env_len,
            frames=frames,
            noise_sigma=env_noise,
            baseline=env_base,
            amp_scale=env_scale,
            sources=parse_sources(src_text),
        ),
        kernel=dict(
            enabled=k_enabled,
            radius=k_radius,
            lr=k_lr,
            l2=k_l2,
            init=k_init,
            k_gate=k_gate,
        ),
    )


def run_live():
    user_cfg = build_user_cfg_dict()
    engine = Engine(make_config_from_dict(user_cfg))

    last_draw = -1

    def redraw(t):
        draw_combined_heatmap(combo_ph, engine.env, engine.S, t_now=t, window=view_window)
        draw_energy(energy_ph, engine.hist)
        draw_kernel(engine.w, kernel_ph)

    def cb(t):
        nonlocal last_draw
        if (t - last_draw) >= int(chunk) or t == engine.T - 1:
            last_draw = t
            redraw(t)

    engine.run(progress_cb=cb, snapshot_every=int(chunk))
    redraw(engine.T - 1)

    with diag:
        st.caption(f"Done. frames={engine.T}, space={engine.X}, env.length={engine.env.shape[1]}  |  boundary offset={engine.offset:.2f}")
        st.progress(1.0)


if run_btn:
    run_live()