# core/smooth_plot_helpers.py
# Plotly helpers for a single combined (overlay) heatmap and a live energy chart.

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

try:
    import streamlit as st
except Exception:
    st = None  # allow import without Streamlit (for tests)


# -----------------------
# Utilities
# -----------------------

def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.size == 0:
        return a
    amin = np.nanmin(a)
    amax = np.nanmax(a)
    rng = amax - amin
    if not np.isfinite(rng) or rng == 0:
        return np.zeros_like(a)
    return (a - amin) / (rng + 1e-12)


def _dark_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title="x (space)",
        yaxis_title="t (time)",
        legend=dict(orientation="h", y=1.02, x=0.0),
        uirevision="keep",  # keep zoom on update
    )
    return fig


# -----------------------
# Combined heatmap helpers
# -----------------------

def ensure_combo_fig(
    ss_key: str,
    T: int,
    X_env: int,
    X_space: int,
    height: int = 520,
    **kwargs,  # swallow extras like title to avoid TypeError
) -> go.Figure:
    """
    Ensure a combined overlay figure exists in st.session_state[ss_key].
    Returns the figure. If shape changes, a fresh one is created.
    """
    title = kwargs.get("title", "Environment + Substrate (t vs x)")

    if st is None:
        # no Streamlit context; just build a new fig each call
        return _make_combo_fig(T, X_env, X_space, height, title)

    fig = st.session_state.get(ss_key)
    if not isinstance(fig, go.Figure):
        fig = _make_combo_fig(T, X_env, X_space, height, title)
        st.session_state[ss_key] = fig
        return fig

    # Validate traces / shapes; rebuild if mismatched
    try:
        ok = (
            len(fig.data) == 2
            and isinstance(fig.data[0], go.Heatmap)
            and isinstance(fig.data[1], go.Heatmap)
            and fig.data[0].z.shape == (T, X_env)
            and fig.data[1].z.shape == (T, X_space)
        )
    except Exception:
        ok = False

    if not ok:
        fig = _make_combo_fig(T, X_env, X_space, height, title)
        st.session_state[ss_key] = fig

    return fig


def _make_combo_fig(T: int, X_env: int, X_space: int, height: int, title: str) -> go.Figure:
    # Empty (zeros) normalized arrays to start
    Zenv = np.zeros((T, X_env), dtype=float)
    Zsub = np.zeros((T, X_space), dtype=float)

    fig = _dark_fig(title)
    # Overlay both heatmaps on the same axes; use opacity + distinct colorscales
    fig.add_trace(go.Heatmap(
        z=Zenv,
        colorscale="Viridis",
        zsmooth=False,
        showscale=True,
        colorbar=dict(title="Env", x=1.02),
        name="Env",
        opacity=0.70,
    ))
    fig.add_trace(go.Heatmap(
        z=Zsub,
        colorscale="Inferno",
        zsmooth=False,
        showscale=True,
        colorbar=dict(title="Substrate", x=1.08),
        name="Substrate",
        opacity=0.55,
    ))
    fig.update_layout(height=height)
    return fig


def update_combo_fig(fig: go.Figure, env_full: np.ndarray, subs_full: np.ndarray) -> go.Figure:
    """
    In-place update of the two heatmap traces with normalized slices.
    env_full : [t, X_env], subs_full : [t, X_space], where t ≤ T.
    """
    Ze = _normalize(env_full)
    Zs = _normalize(subs_full)

    # pad to full T with zeros (so array shape stays constant)
    T_full, X_env = fig.data[0].z.shape
    t = Ze.shape[0]
    if t < T_full:
        pad_env = np.zeros((T_full - t, Ze.shape[1]), dtype=float)
        Ze = np.vstack([Ze, pad_env])

    T_full2, X_sub = fig.data[1].z.shape
    t2 = Zs.shape[0]
    if t2 < T_full2:
        pad_sub = np.zeros((T_full2 - t2, Zs.shape[1]), dtype=float)
        Zs = np.vstack([Zs, pad_sub])

    fig.data[0].z = Ze
    fig.data[1].z = Zs
    return fig


# -----------------------
# Energy timeseries (live)
# -----------------------

def draw_energy_timeseries_live(placeholder, t, e_cell, e_env, e_flux, title: str = "Energy vs time"):
    """
    Overwrite placeholder with a dark themed, live-updating line chart.
    """
    fig = _dark_fig(title)
    fig.update_layout(height=280, xaxis_title="t (frames)", yaxis_title="energy")

    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux", mode="lines"))

    placeholder.plotly_chart(fig, use_container_width=True, theme=None)