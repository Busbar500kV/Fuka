# smooth_plot_helpers.py
# Helpers for smooth, live Plotly updates (no blinking) and a combined heatmap.

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Small utilities
# -----------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Scale to [0,1] robustly. Works on 1D or 2D arrays."""
    if arr.size == 0:
        return arr
    a_min = np.nanmin(arr)
    a_max = np.nanmax(arr)
    ptp = a_max - a_min
    if ptp <= 1e-12:
        return np.zeros_like(arr)
    return (arr - a_min) / (ptp + 1e-12)

def _resample_width(A: np.ndarray, new_w: int) -> np.ndarray:
    """
    Resample 2D data [T, X] to width new_w using integer index mapping.
    Keeps periodic feel and is super fast.
    """
    if A.shape[1] == new_w:
        return A
    idx = (np.arange(new_w) * A.shape[1] // new_w) % A.shape[1]
    return A[:, idx]

# -----------------------------
# Combined heatmap (Env + Substrate) overlaid on SAME axes
# -----------------------------

def make_combo_fig(
    T: int,
    X_env: int,
    X_space: int,
    height: int = 520,
    title: str = "Environment + Substrate (t vs x)",
    env_colorscale: str = "Viridis",
    subs_colorscale: str = "Inferno",
    env_opacity: float = 0.65,
    subs_opacity: float = 0.65,
) -> go.Figure:
    """
    Build a single figure with TWO Heatmap traces sharing the same axes:
      - trace[0]: Env (normalized)
      - trace[1]: Substrate (normalized, resampled to env width)
    We initialize with tiny arrays; you'll update the z-data live.
    """
    # Seed tiny data so the traces exist (so later we can update .data[i].z)
    En0 = np.zeros((2, 2))
    Sn0 = np.zeros((2, 2))

    fig = go.Figure()

    # Env heatmap
    fig.add_trace(go.Heatmap(
        z=En0,
        colorscale=env_colorscale,
        showscale=True,
        name="Env",
        opacity=env_opacity,
        zsmooth=False,
        colorbar=dict(title="Env")
    ))

    # Substrate heatmap (overlay on same axes)
    fig.add_trace(go.Heatmap(
        z=Sn0,
        colorscale=subs_colorscale,
        showscale=True,
        name="Substrate",
        opacity=subs_opacity,
        zsmooth=False,
        colorbar=dict(title="Substrate")
    ))

    # Layout: dark theme, fixed paper/plot bg, keep user zoom via uirevision
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        xaxis=dict(
            title="x (space)",
            showgrid=False,
            zeroline=False,
            color="#CCCCCC"
        ),
        yaxis=dict(
            title="t (time)",
            autorange="reversed",  # time increases downward to feel like a raster
            showgrid=False,
            zeroline=False,
            color="#CCCCCC"
        ),
        font=dict(color="#DDDDDD"),
        uirevision="keep-zoom"  # <-- critical to prevent re-zoom on updates
    )

    return fig

def update_combo_fig(
    fig: go.Figure,
    env_partial: np.ndarray,     # shape [t_used, X_env]
    subs_partial: np.ndarray,    # shape [t_used, X_space]
) -> go.Figure:
    """
    Update the two heatmap traces' z data IN-PLACE.
    - Normalizes each to [0,1].
    - Resamples substrate to env width so they overlay 1:1 in x.
    """
    # Sanity: empty arrays? Keep fig as-is.
    if env_partial.size == 0 or subs_partial.size == 0:
        return fig

    T_used = env_partial.shape[0]
    X_env  = env_partial.shape[1]

    En = _normalize(env_partial)
    # resample substrate to X_env
    subs_rs = _resample_width(subs_partial, X_env)
    Sn = _normalize(subs_rs)

    # Ensure both traces exist
    if len(fig.data) < 2:
        # Rebuild if someone gave us a mismatched fig
        fig = make_combo_fig(T_used, X_env, subs_partial.shape[1])

    # Update z for both traces
    fig.data[0].z = En
    fig.data[1].z = Sn

    # Keep axis ranges stable while allowing user zoom (uirevision)
    fig.update_xaxes(range=[0, X_env - 1])
    fig.update_yaxes(range=[0, T_used - 1])

    return fig

# -----------------------------
# Session-state helpers (no blinking)
# -----------------------------

def ensure_combo_fig(
    ss_key: str,
    T: int,
    X_env: int,
    X_space: int,
    height: int = 520,
    title: str = "Environment + Substrate (t vs x)",
) -> go.Figure:
    """
    Make once and cache in Streamlit session_state so subsequent updates
    don't recreate the figure object (prevents flicker + resets).
    """
    import streamlit as st  # local import to avoid hard dependency for unit tests
    if ss_key not in st.session_state:
        st.session_state[ss_key] = make_combo_fig(
            T=T, X_env=X_env, X_space=X_space, height=height, title=title
        )
    return st.session_state[ss_key]

# -----------------------------
# Energy time-series (live)
# -----------------------------

def ensure_energy_fig(ss_key: str, title: str = "Energy vs time") -> go.Figure:
    """
    Create (once) a line figure with three series: E_cell, E_env, E_flux.
    """
    import streamlit as st
    if ss_key in st.session_state:
        return st.session_state[ss_key]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[], mode="lines", name="E_cell"))
    fig.add_trace(go.Scatter(y=[], mode="lines", name="E_env"))
    fig.add_trace(go.Scatter(y=[], mode="lines", name="E_flux"))

    fig.update_layout(
        title=title,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        xaxis=dict(title="t", showgrid=False, color="#CCCCCC"),
        yaxis=dict(title="Energy", showgrid=True, gridcolor="#333333", color="#CCCCCC"),
        font=dict(color="#DDDDDD"),
        uirevision="keep-zoom"
    )
    st.session_state[ss_key] = fig
    return fig

def draw_energy_timeseries_live(
    placeholder,
    t: Sequence[int],
    e_cell: Sequence[float],
    e_env: Sequence[float],
    e_flux: Sequence[float],
    title: str = "Energy vs time",
    ss_key: str = "energy_fig",
):
    """
    Update energy figure in place (no blinking).
    Pass entire history arrays each time; we only mutate y-data.
    """
    fig = ensure_energy_fig(ss_key, title=title)

    # Ensure traces exist (defensive)
    while len(fig.data) < 3:
        fig.add_trace(go.Scatter(y=[], mode="lines"))

    fig.data[0].y = list(e_cell)
    fig.data[1].y = list(e_env)
    fig.data[2].y = list(e_flux)

    # X uses implicit 0..len-1; if you want explicit t on x, set x arrays too:
    # fig.data[0].x = fig.data[1].x = fig.data[2].x = list(t)

    placeholder.plotly_chart(fig, use_container_width=True, theme=None)