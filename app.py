# app.py
# Streamlit UI — live simulation with energy chart, env/substrate heatmaps,
# combined view (stacked/overlay), and kernel panel.
import json
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sim_core import make_engine, default_config, Engine


st.set_page_config(page_title="Fuka — Free‑Energy Gradient (Live)", layout="wide")

# ---------------- UI helpers ----------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_energy(hist_placeholder, hist):
    with hist_placeholder:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.plot(hist.t, hist.E_cell, label="E_cell")
        ax.plot(hist.t, hist.E_env,  label="E_env")
        ax.plot(hist.t, hist.E_flux, label="E_flux")
        ax.set_title("Energies (live)")
        ax.legend(loc="upper left")
        ax.set_xlabel("frame")
        st.pyplot(fig, clear_figure=True)

def draw_heatmaps(env_ph, subs_ph, env, S):
    with env_ph:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        im = ax.imshow(env.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Environment E(t,x)")
        ax.set_xlabel("frame"); ax.set_ylabel("space")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, clear_figure=True)
    with subs_ph:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        im = ax.imshow(S.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Substrate S(t,x)")
        ax.set_xlabel("frame"); ax.set_ylabel("space")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, clear_figure=True)

def draw_combined(combo_ph, env, S, mode="stacked", alpha=0.35):
    with combo_ph:
        if mode == "overlay":
            # rescale both to [0,1] for a meaningful overlay
            e = env.copy()
            s = S.copy()
            e = (e - e.min()) / (e.ptp() + 1e-12)
            s = (s - s.min()) / (s.ptp() + 1e-12)
            rgb = np.zeros((e.shape[0], e.shape[1], 3), dtype=float)
            # env in green, substrate in red
            rgb[:, :, 1] = e
            rgb[:, :, 0] = s
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
            ax.imshow(rgb.transpose(1,0,2), origin="lower", aspect="auto")
            ax.set_title("Combined view (overlay: red=S, green=E)")
            ax.set_xlabel("frame"); ax.set_ylabel("space")
            st.pyplot(fig, clear_figure=True)
        else:
            # stacked
            fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
            im0 = axs[0].imshow(env.T, origin="lower", aspect="auto", cmap="viridis")
            axs[0].set_title("Environment E(t,x)")
            axs[0].set_ylabel("space"); fig.colorbar(im0, ax=axs[0])
            im1 = axs[1].imshow(S.T, origin="lower", aspect="auto", cmap="viridis")
            axs[1].set_title("Substrate S(t,x)")
            axs[1].set_xlabel("frame"); axs[1].set_ylabel("space"); fig.colorbar(im1, ax=axs[1])
            st.pyplot(fig, clear_figure=True)

def draw_kernels(kph, engine: Engine, max_show=6):
    with kph:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        K = engine.K
        shown = min(max_show, len(engine.w))
        if shown == 0:
            ax.text(0.5, 0.5, "no kernels yet", ha="center", va="center")
        else:
            for i in range(shown):
                wi = engine.w[i]
                ax.plot(np.arange(-K, K+1), wi, alpha=0.8)
        ax.set_title("Gate kernels (sample)")
        ax.set_xlabel("offset")
        ax.set_ylabel("weight")
        st.pyplot(fig, clear_figure=True)


# ---------------- Sidebar controls ----------------

cfg = default_config()
with st.sidebar:
    st.title("Controls")

    seed   = st.number_input("Seed", 0, 10000, value=int(cfg["seed"]))
    frames = st.number_input("Frames", 100, 50000, value=int(cfg["frames"]), step=500)
    space  = st.number_input("Space (cells)", 16, 512, value=int(cfg["space"]))
    band   = st.number_input("band (boundary width)", 1, 8, value=int(cfg["band"]))

    k_flux  = st.number_input("k_flux (boundary flux)", 0.0, 2.0, value=float(cfg["k_flux"]),  step=0.01)
    k_motor = st.number_input("k_motor (motor explore)", 0.0, 10.0, value=float(cfg["k_motor"]), step=0.01)
    motor_noise = st.number_input("motor_noise", 0.0, 1.0, value=float(cfg["motor_noise"]), step=0.01)
    c_motor = st.number_input("c_motor (motor work cost)", 0.0, 5.0, value=float(cfg["c_motor"]), step=0.01)
    alpha_move = st.number_input("alpha_move (motion gain)", 0.0, 2.0, value=float(cfg["alpha_move"]), step=0.01)
    beta_tension = st.number_input("beta_tension (tension)", 0.0, 1.0, value=float(cfg["beta_tension"]), step=0.01)

    decay   = st.number_input("decay",   0.0, 0.5, value=float(cfg["decay"]),   step=0.001)
    diffuse = st.number_input("diffuse", 0.0, 1.0, value=float(cfg["diffuse"]), step=0.01)

    gate_win = st.number_input("gate_win (half width K)", 1, 64, value=int(cfg["gate_win"]))
    eta      = st.number_input("eta (kernel LR)", 0.000, 1.0, value=float(cfg["eta"]), step=0.001, format="%.3f")
    ema_beta = st.number_input("ema_beta (baseline EMA)", 0.000, 1.0, value=float(cfg["ema_beta"]), step=0.01)
    lam_l1   = st.number_input("lam_l1 (L1 shrink)", 0.00, 1.00, value=float(cfg["lam_l1"]), step=0.01)
    prune_th = st.number_input("prune_thresh", 0.00, 1.00, value=float(cfg["prune_thresh"]), step=0.01)
    min_age  = st.number_input("min_age", 0, 100000, value=int(cfg["min_age"]), step=50)
    spawn_rate = st.number_input("spawn_rate", 0.0, 1.0, value=float(cfg["spawn_rate"]), step=0.01)
    max_conns  = st.number_input("max_conns", 1, 1024, value=int(cfg["max_conns"]))

    env_len  = st.number_input("Env length", 32, 4096, value=int(cfg["env"]["length"]))
    env_sig  = st.number_input("Env noise σ", 0.0, 1.0, value=float(cfg["env"]["noise_sigma"]), step=0.01)

    st.caption("Edit env sources JSON")
    sources_str = st.text_area("Sources JSON", value=json.dumps(cfg["env"]["sources"], indent=2), height=180)
    try:
        sources_ok = json.loads(sources_str)
        st.success("Sources OK")
    except Exception as e:
        st.error(f"JSON error: {e}")
        sources_ok = cfg["env"]["sources"]

    live = st.checkbox("Live streaming", value=True)
    chunk = int(st.slider("Refresh chunk (frames)", 20, 500, 150))

    combo_mode = st.selectbox("Combined view mode", options=["stacked", "overlay"], index=0)

user_cfg = {
    "seed": seed, "frames": frames, "space": space, "band": band,
    "k_flux": k_flux, "k_motor": k_motor, "motor_noise": motor_noise,
    "c_motor": c_motor, "alpha_move": alpha_move, "beta_tension": beta_tension,
    "decay": decay, "diffuse": diffuse,
    "gate_win": gate_win, "eta": eta, "ema_beta": ema_beta,
    "lam_l1": lam_l1, "prune_thresh": prune_th, "min_age": min_age,
    "spawn_rate": spawn_rate, "max_conns": max_conns,
    "env": {"length": env_len, "noise_sigma": env_sig, "frames": frames, "sources": sources_ok},
}

# ---------------- Main layout ----------------
st.title("Fuka — Free‑Energy Gradient Simulation (Live)")
status = st.empty()
energy_ph = st.empty()

colA, colB = st.columns(2)
env_ph = colA.empty()
subs_ph = colB.empty()

combo_ph = st.container()  # combined view always shown after the two heatmaps
kern_ph = st.container()

# ---------------- Run button ----------------
def run_live():
    engine = make_engine(user_cfg)
    env = engine.env  # fixed
    S = engine.S      # we will fill as it runs

    status.info("Starting…")

    # initial draws
    draw_energy(energy_ph, engine.hist)
    draw_heatmaps(env_ph, subs_ph, env, S)
    draw_combined(combo_ph, env, S, mode=combo_mode)
    draw_kernels(kern_ph, engine)

    last = [-1]  # mutable closure cell

    def cb(t: int):
        if t - last[0] >= chunk or t == engine.T - 1:
            last[0] = t
            draw_energy(energy_ph, engine.hist)
            draw_heatmaps(env_ph, subs_ph, env, engine.S)
            draw_combined(combo_ph, env, engine.S, mode=combo_mode)
            draw_kernels(kern_ph, engine)
            status.success(f"Running… frame {t+1}/{engine.T}")

    engine.run(progress_cb=cb, snapshot_every=int(chunk))
    status.success("Done!")

if st.button("Run / Rerun", use_container_width=True):
    run_live()