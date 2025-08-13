import streamlit as st
from plots_full import streamlit_live_stacked  # or streamlit_live_overlay

# somewhere near your UI creation:
combo_placeholder = st.empty()

# during the run loop, on each chunk:
def get_arrays():
    # engine.env_full: (frames, env_len)
    # engine.S_full:   (frames, space)
    return engine.env_full, engine.S_full

streamlit_live_stacked(st, combo_placeholder, get_arrays, width=engine.env_full.shape[1], title="E(t,x) • S(t,x)")