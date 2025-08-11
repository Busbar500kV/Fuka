# app.py — Streamlit dashboard (live)
import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sim_core import make_engine, default_config, Engine

st.set_page_config(page_title="Fuka – Free‑Energy Gradient Simulation (Live)", layout="wide")

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Controls")

# Keep a dict in session_state so reruns preserve your settings
if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()
cfg = st.session_state.cfg

seed    = st.sidebar.number_input("Seed", 0, 10_000, int(cfg["seed"]), step=1)
frames  = st.sidebar.number_input("Frames", 100, 100_000, int(cfg["frames"]), step=100)
space   = st.sidebar.number_input("Space (cells)", 8, 1_024, int(cfg["space"]), step=1)

k_flux  = st.sidebar.number_input("k_flux (boundary flux)", 0.0, 10.0, float(cfg.get("k_flux", 0.08)), step=0.01, format="%.3f")
k_motor = st.sidebar.number_input("k_motor (motor explore)", 0.0, 20.0, float(cfg.get("k_motor", 4.90)), step=0.10, format="%.2f")
k_noise = st.sidebar.number_input("k_noise (band noise)", 0.0, 2.0, float(cfg.get("k_noise", 0.02)), step=0.01, format="%.2f")

decay   = st.sidebar.number_input("decay", 0.0, 1.0, float(cfg.get("decay", 0.01)), step=0.005, format="%.3f")
diffuse = st.sidebar.number_input("diffuse", 0.0, 2.0, float(cfg.get("diffuse", 0.15)), step=0.01, format="%.2f")
band    = st.sidebar.number_input("band (boundary width)", 1, 32, int(cfg.get("band", 3)), step=1)

c_motor = st.sidebar.number_input("c_motor (motor work cost)", 0.0, 1.0, float(cfg.get("c_motor", 0.02)), step=0.005, format="%.3f")

# Environment section
st.sidebar.subheader("Environment")
env_len  = st.sidebar.number_input("Env length", 8, 4096, int(cfg["env"]["length"]), step=8)
env_sig  = st.sidebar.number_input("Env noise σ", 0.0, 1.0, float(cfg["env"]["noise_sigma"]), step=0.01)

st.sidebar.markdown("**Sources JSON (list)**")
src_str = st.sidebar.text_area(
    "Edit env sources",
    value=json.dumps(cfg["env"]["sources"], indent=2),
    height=220
)
sources_ok = True
try:
    src_list = json.loads(src_str)
    assert isinstance(src_list, list)
    st.sidebar.success("Sources OK")
except Exception as e:
    sources_ok = False
    st.sidebar.error(f"Invalid JSON: {e}")
    src_list = cfg["env"]["sources"]

live_mode = st.sidebar.checkbox("Live streaming", value=True)
chunk     = st.sidebar.slider("Refresh chunk (frames)", 10, 500, 120)

# Update session cfg dict
cfg.update(dict(
    seed=seed, frames=frames, space=space,
    k_flux=k_flux, k_motor=k_motor, k_noise=k_noise,
    decay=decay, diffuse=diffuse, band=band, c_motor=c_motor,
))
cfg["env"].update(dict(length=env_len, frames=frames, noise_sigma=env_sig, sources=src_list))

# --------------------------
# Main layout
# --------------------------
st.title("Fuka – Free‑Energy Gradient Simulation (Live)")
status = st.empty()

# Live line chart for energies
fig_energy, ax_energy = plt.subplots()
line_cell, = ax_energy.plot([], [], label="E_cell")
line_env,  = ax_energy.plot([], [], label="E_env")
line_flux, = ax_energy.plot([], [], label="E_flux")
ax_energy.set_title("Energies (live)")
ax_energy.legend(loc="upper left")
energy_ph = st.pyplot(fig_energy, clear_figure=False)

# Live heatmaps (environment and substrate)
col1, col2 = st.columns(2)
with col1:
    fig_env, ax_env = plt.subplots()
    env_im = ax_env.imshow(np.zeros((10, 10)), aspect="auto", origin="lower")
    ax_env.set_title("Environment E(t,x)")
    ax_env.set_xlabel("frame"); ax_env.set_ylabel("space")
    env_ph = st.pyplot(fig_env, clear_figure=False)

with col2:
    fig_sub, ax_sub = plt.subplots()
    sub_im = ax_sub.imshow(np.zeros((10, 10)), aspect="auto", origin="lower")
    ax_sub.set_title("Substrate S(t,x)")
    ax_sub.set_xlabel("frame"); ax_sub.set_ylabel("space")
    sub_ph = st.pyplot(fig_sub, clear_figure=False)

# --------------------------
# Run button
# --------------------------
if st.button("Run / Rerun"):
    status.info("Starting…")

    # Build a fresh engine
    engine: Engine = make_engine(cfg)

    # Buffers for live plots
    t_list, c_list, e_list, f_list = [], [], [], []

    # To build heatmaps incrementally we pre-allocate
    S_live = np.zeros((engine.T, engine.X), dtype=float)
    # We’ll construct a resampled env heatmap on the fly
    E_live = np.zeros((engine.T, engine.X), dtype=float)

    def redraw(t: int):
        # Update lines
        line_cell.set_data(t_list, c_list)
        line_env.set_data(t_list, e_list)
        line_flux.set_data(t_list, f_list)
        ax_energy.relim(); ax_energy.autoscale_view()
        energy_ph.pyplot(fig_energy, clear_figure=False)

        # Update env heatmap (show all rows up to current t)
        env_im.set_data(E_live[:t+1].T)
        env_im.set_extent([0, t+1, 0, engine.X])  # frame axis grows
        fig_env.tight_layout()
        env_ph.pyplot(fig_env, clear_figure=False)

        # Update substrate heatmap
        sub_im.set_data(S_live[:t+1].T)
        sub_im.set_extent([0, t+1, 0, engine.X])
        fig_sub.tight_layout()
        sub_ph.pyplot(fig_sub, clear_figure=False)

    if live_mode:
        last = [-1]  # mutable closure storage

        def cb(t: int):
            # pull last values recorded by the engine
            t_list.append(t)
            c_list.append(engine.hist.E_cell[-1])
            e_list.append(engine.hist.E_env[-1])
            f_list.append(engine.hist.E_flux[-1])

            # fill the images
            S_live[t] = engine.S[t]
            # env slice resampled used inside engine.step; reconstruct it here too
            # (call private helper is okay for app convenience)
            E_live[t] = engine._env_row_resampled(t)  # noqa

            # redraw every chunk or at end
            if (t - last[0] >= chunk) or (t == engine.T - 1):
                last[0] = t
                redraw(t)

        engine.run(progress_cb=cb)
        status.success("Done!")
    else:
        engine.run(progress_cb=None)
        # Fill buffers for a final render
        t_list = engine.hist.t
        c_list = engine.hist.E_cell
        e_list = engine.hist.E_env
        f_list = engine.hist.E_flux
        S_live[:, :] = engine.S
        for t in range(engine.T):
            E_live[t] = engine._env_row_resampled(t)  # noqa
        redraw(engine.T - 1)
        status.success("Done!")