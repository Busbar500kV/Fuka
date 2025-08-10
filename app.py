# --- app.py ---
import streamlit as st
from sim_core import run_sim, default_config
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="FUKA", layout="wide")
st.title("First Universal Common Ancestor (FUKA) – demo")

cfg = default_config()
col1, col2, col3 = st.columns(3)
cfg["frames"] = col1.number_input("Frames", 100, 5000, cfg["frames"], 100)
cfg["space"]  = col2.number_input("Space", 16, 256, cfg["space"], 16)
cfg["seed"]   = col3.number_input("Seed", 0, 10_000, cfg["seed"], 1)

if st.button("Run simulation"):
    out = run_sim(cfg)

    st.subheader("Energy")
    st.write(float(out["energy"]))

    st.subheader("Environment (X×T)")
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(out["env"], aspect="auto", origin="lower")
    st.pyplot(fig1)

    st.subheader("Substrate (X×T)")
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(out["subs"], aspect="auto", origin="lower")
    st.pyplot(fig2)
else:
    st.info("Set parameters and click **Run simulation**.")