import streamlit as st
from sim_core import run_sim, default_config
import numpy as np
import pandas as pd


# --------------- sidebar controls ---------------
with st.sidebar:
    st.header("Run controls")

    def clamp(v, lo, hi): 
        return int(max(lo, min(hi, int(v))))

    # guard against stale/session values
    frames_min, frames_max, frames_step = 400, 10000, 200
    space_min,  space_max               = 64, 512

    cfg["frames"] = clamp(cfg.get("frames", 1600), frames_min, frames_max)
    cfg["space"]  = clamp(cfg.get("space", 192),   space_min,  space_max)
    cfg["seed"]   = int(cfg.get("seed", 0))

    cfg["frames"] = st.number_input(
        "Frames", min_value=frames_min, max_value=frames_max,
        value=cfg["frames"], step=frames_step, key="frames_num"
    )
    cfg["space"] = st.number_input(
        "Space", min_value=space_min, max_value=space_max,
        value=cfg["space"], step=32, key="space_num"
    )
    cfg["seed"] = st.number_input(
        "Seed", min_value=0, max_value=999_999,
        value=cfg["seed"], step=1, key="seed_num"
    )

    st.header("Environment")
    cfg["env_baseline"] = float(st.slider("Baseline free energy", 0.0, 1.0, float(cfg["env_baseline"]), 0.01, key="env_base"))
    cfg["env_pulse_rate"] = float(st.slider("Pulse spawn rate", 0.0, 0.05, float(cfg["env_pulse_rate"]), 0.001, key="env_rate"))
    cfg["env_pulse_E"] = float(st.slider("Pulse energy", 0.0, 20.0, float(cfg["env_pulse_E"]), 0.5, key="env_E"))
    cfg["env_boundary_boost"] = float(st.slider("Boundary boost (motors!)", 0.0, 5.0, float(cfg["env_boundary_boost"]), 0.1, key="env_boost"))

    st.header("Substrate")
    cfg["decay_subs"]   = float(st.slider("Decay", 0.0, 0.1, float(cfg["decay_subs"]), 0.001, key="decay"))
    cfg["diffuse_subs"] = float(st.slider("Diffusion", 0.0, 0.49, float(cfg["diffuse_subs"]), 0.01, key="diffuse"))

    st.header("Connections")
    cfg["init_n"]      = int(st.number_input("Initial connections", 1, 64, int(cfg["init_n"]), 1, key="init_n"))
    cfg["grow_every"]  = int(st.number_input("Grow every (frames)", 50, 2000, int(cfg["grow_every"]), 50, key="grow_every"))
    cfg["grow_budget"] = int(st.number_input("Growth attempts", 1, 20, int(cfg["grow_budget"]), 1, key="grow_budget"))
    cfg["prune_every"] = int(st.number_input("Prune every (frames)", 50, 2000, int(cfg["prune_every"]), 50, key="prune_every"))

    st.header("Weights & costs")
    c = st.columns(3)
    cfg["gamma_direct"]   = float(c[0].slider("γ_direct", 0.0, 3.0, float(cfg["gamma_direct"]), 0.1, key="gdir"))
    cfg["gamma_indirect"] = float(c[1].slider("γ_indirect", 0.0, 3.0, float(cfg["gamma_indirect"]), 0.1, key="gind"))
    cfg["gamma_motor"]    = float(c[2].slider("γ_motor", 0.0, 3.0, float(cfg["gamma_motor"]), 0.1, key="gmot"))

    c = st.columns(3)
    cfg["cost_activation"]  = float(c[0].slider("C_activation", 0.0, 0.2, float(cfg["cost_activation"]), 0.005, key="cact"))
    cfg["cost_maintenance"] = float(c[1].slider("C_maintenance", 0.0, 0.2, float(cfg["cost_maintenance"]), 0.005, key="cmain"))
    cfg["cost_plasticity"]  = float(c[2].slider("C_plasticity (reserved)", 0.0, 0.1, float(cfg["cost_plasticity"]), 0.005, key="cplast"))