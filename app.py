# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim_core import run_sim, default_config

st.set_page_config(page_title="FUKA — emergent metabolism", layout="wide")
st.title("FUKA — Emergent free‑energy seeking (baseline)")

cfg = default_config()

# --------------- sidebar controls ---------------
with st.sidebar:
    st.header("Run controls")
    cfg["frames"] = st.number_input("Frames", 400, 10000, cfg["frames"], 200)
    cfg["space"]  = st.number_input("Space", 64, 512, cfg["space"], 32)
    cfg["seed"]   = st.number_input("Seed", 0, 999999, cfg["seed"], 1)

    st.header("Environment")
    cfg["env_baseline"]       = st.slider("Baseline free energy", 0.0, 1.0, cfg["env_baseline"], 0.01)
    cfg["env_pulse_rate"]     = st.slider("Pulse spawn rate", 0.0, 0.05, cfg["env_pulse_rate"], 0.001)
    cfg["env_pulse_E"]        = st.slider("Pulse energy", 0.0, 20.0, cfg["env_pulse_E"], 0.5)
    cfg["env_boundary_boost"] = st.slider("Boundary boost (motors!)", 0.0, 5.0, cfg["env_boundary_boost"], 0.1)

    st.header("Substrate")
    cfg["decay_subs"]    = st.slider("Decay", 0.0, 0.1, cfg["decay_subs"], 0.001)
    cfg["diffuse_subs"]  = st.slider("Diffusion", 0.0, 0.49, cfg["diffuse_subs"], 0.01)

    st.header("Connections")
    cfg["init_n"]      = st.number_input("Initial connections", 1, 64, cfg["init_n"], 1)
    cfg["grow_every"]  = st.number_input("Grow every (frames)", 50, 2000, cfg["grow_every"], 50)
    cfg["grow_budget"] = st.number_input("Growth attempts", 1, 20, cfg["grow_budget"], 1)
    cfg["prune_every"] = st.number_input("Prune every (frames)", 50, 2000, cfg["prune_every"], 50)

    st.header("Weights & costs")
    cols = st.columns(3)
    cfg["gamma_direct"]   = cols[0].slider("γ_direct", 0.0, 3.0, cfg["gamma_direct"], 0.1)
    cfg["gamma_indirect"] = cols[1].slider("γ_indirect", 0.0, 3.0, cfg["gamma_indirect"], 0.1)
    cfg["gamma_motor"]    = cols[2].slider("γ_motor", 0.0, 3.0, cfg["gamma_motor"], 0.1)

    cols = st.columns(3)
    cfg["cost_activation"]  = cols[0].slider("C_activation", 0.0, 0.2, cfg["cost_activation"], 0.005)
    cfg["cost_maintenance"] = cols[1].slider("C_maintenance", 0.0, 0.2, cfg["cost_maintenance"], 0.005)
    cfg["cost_plasticity"]  = cols[2].slider("C_plasticity (reserved)", 0.0, 0.1, cfg["cost_plasticity"], 0.005)

run_btn = st.button("Run simulation", use_container_width=True)
if not run_btn:
    st.info("Adjust the parameters on the left and press **Run simulation**.")
    st.stop()

# --------------- run ---------------
out = run_sim(cfg)

# --------------- plots ---------------
X, T = out["env"].shape

c1, c2 = st.columns(2, gap="large")

with c1:
    st.subheader("Environment events (X×T)")
    fig, ax = plt.subplots()
    ax.imshow(out["env"], aspect="auto", origin="lower")
    st.pyplot(fig)

with c2:
    st.subheader("Substrate state S (X×T)")
    fig, ax = plt.subplots()
    ax.imshow(out["subs"], aspect="auto", origin="lower")
    st.pyplot(fig)

st.subheader("Connection activity (rows = connections)")
fig, ax = plt.subplots()
ax.imshow(out["acts"], aspect="auto", origin="lower")
st.pyplot(fig)

st.subheader("Global energy (final)")
st.write(f"{out['energy']:.3f}")

st.subheader("Emergent roles (by dominant harvest)")
role_counts = out["roles"]
fig, ax = plt.subplots()
labels = ["SENSE", "INTERNAL", "MOTOR", "UNLABELED"]
vals = [role_counts.get(k, 0) for k in labels]
ax.bar(labels, vals)
st.pyplot(fig)

st.subheader("Connections table")
df = pd.DataFrame(out["table"])
st.dataframe(df, use_container_width=True)