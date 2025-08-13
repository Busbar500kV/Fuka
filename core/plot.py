# core/plot.py
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

# ---------- small helpers ----------

def _normalize(z: np.ndarray) -> np.ndarray:
    """Safe [0,1] normalize for plotting; handles constants and NaNs."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).any():
        return np.zeros_like(z)
    zmin = np.nanmin(z)
    zmax = np.nanmax(z)
    rng = zmax - zmin
    if rng <= 1e-12:
        return np.zeros_like(z)
    return (z - zmin) / (rng + 1e-12)

def _fig_dark():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=10, t=40, b=40),
        height=320,
        legend=dict(orientation="h", y=1.02, x=0.0),
    )
    return fig

# ---------- public: energy timeseries ----------

def draw_energy_timeseries(placeholder, t, e_cell, e_env, e_flux, title="Energy vs time"):
    """
    Overwrites the placeholder with a single up-to-date time series figure.
    Inputs:
      - t: 1D array/list of ints
      - e_cell, e_env, e_flux: 1D arrays/lists same length as t
    """
    fig = _fig_dark()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux", mode="lines"))
    fig.update_layout(title=title, xaxis_title="t (frame)", yaxis_title="Energy")
    placeholder.plotly_chart(fig, use_container_width=True)

# ---------- public: combined overlay (last frame) ----------

def draw_overlay_last_frame(placeholder, env_row, subs_row, title="Env + Substrate (last frame)"):
    """
    Draw a single combined line plot for the last-frame slice:
      env_row: 1D array of environment at t = last
      subs_row: 1D array of substrate at t = last (same length as env_row after any resample)
    """
    x = np.arange(len(env_row))
    fig = _fig_dark()
    fig.add_trace(go.Scatter(x=x, y=env_row, name="Env", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=subs_row, name="Substrate", mode="lines"))
    fig.update_layout(title=title, xaxis_title="x (space index)", yaxis_title="value")
    placeholder.plotly_chart(fig, use_container_width=True)

# ---------- public: full heatmaps (env + substrate) ----------

def draw_heatmap_full(placeholder, env_full, subs_full, title="E(t,x) • S(t,x)"):
    """
    Overwrites the placeholder with a single figure containing two heatmaps
    stacked vertically: top = Env(t,x), bottom = Substrate(t,x).
      env_full  : 2D array [T, X_env]
      subs_full : 2D array [T, X_space]
    Both are normalized to [0,1] for display.
    """
    En = _normalize(env_full)
    Sn = _normalize(subs_full)

    fig = _fig_dark()
    # environment heatmap
    fig.add_trace(go.Heatmap(
        z=En, colorscale="Viridis", colorbar=dict(title="Env", y=0.75), showscale=True,
        zsmooth=False, name="Env"
    ))
    # use a 2nd heatmap via new yaxis
    fig.add_trace(go.Heatmap(
        z=Sn, colorscale="Inferno", colorbar=dict(title="Substrate", y=0.25), showscale=True,
        zsmooth=False, name="Substrate", xaxis="x2", yaxis="y2"
    ))

    # Stack them: create two y-axes sharing the same x index range (by default)
    fig.update_layout(
        title=title,
        height=520,
        xaxis=dict(domain=[0, 1], anchor="y"),
        yaxis=dict(domain=[0.55, 1.0], title="t (Env)"),
        xaxis2=dict(domain=[0, 1], anchor="y2"),
        yaxis2=dict(domain=[0.0, 0.45], title="t (Substrate)"),
    )
    placeholder.plotly_chart(fig, use_container_width=True)