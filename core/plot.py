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

def draw_heatmap_full2(placeholder, env_full, subs_full, title="E(t,x) • S(t,x)"):
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
    # substrate heatmap (second panel)
    fig.add_trace(go.Heatmap(
        z=Sn, colorscale="Inferno", colorbar=dict(title="Substrate", y=0.25), showscale=True,
        zsmooth=False, name="Substrate", xaxis="x2", yaxis="y2"
    ))

    # Stack them vertically
    fig.update_layout(
        title=title,
        height=520,
        xaxis=dict(domain=[0, 1], anchor="y"),
        yaxis=dict(domain=[0.55, 1.0], title="t (Env)"),
        xaxis2=dict(domain=[0, 1], anchor="y2"),
        yaxis2=dict(domain=[0.0, 0.45], title="t (Substrate)"),
    )
    placeholder.plotly_chart(fig, use_container_width=True)
    
    
import numpy as np
import plotly.graph_objects as go

def _normalize_z(z: np.ndarray) -> np.ndarray:
    zmin = np.nanmin(z)
    zptp = np.nanmax(z) - zmin
    if not np.isfinite(zptp) or zptp == 0.0:
        return np.zeros_like(z)
    return (z - zmin) / (zptp + 1e-12)

def _resample_width(arr_2d: np.ndarray, new_w: int) -> np.ndarray:
    """
    Resample a 2D array [T, W_old] to [T, new_w] by 1D interpolation along width.
    Keeps T unchanged. Works for either env or substrate.
    """
    T, W = arr_2d.shape
    if W == new_w:
        return arr_2d
    # original and target x-coordinates in [0,1]
    x_old = np.linspace(0.0, 1.0, W)
    x_new = np.linspace(0.0, 1.0, new_w)
    out = np.empty((T, new_w), dtype=float)
    for t in range(T):
        out[t] = np.interp(x_new, x_old, arr_2d[t])
    return out

def _fig_dark():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=60, t=50, b=40),
        height=520,
    )
    return fig

def draw_heatmap_full(placeholder, env_full: np.ndarray, subs_full: np.ndarray,
                                  title="Combined heatmap: Env ⬤ + Substrate ⬤"):
    """
    Overlay Env(t,x) and Substrate(t,x) on *one* zoomable heatmap (shared axes).
      - Resamples both to X_max = max(env_x, subs_x)
      - Normalizes each independently to [0,1] for visibility
      - Uses two color scales with opacity so both layers are visible
    """
    if env_full.ndim != 2 or subs_full.ndim != 2:
        raise ValueError("env_full and subs_full must be 2D arrays [T, X].")

    T_env, X_env = env_full.shape
    T_sub, X_sub = subs_full.shape
    if T_env != T_sub:
        # If lengths differ a little due to streaming, trim to the shortest
        T = min(T_env, T_sub)
        E = env_full[:T]
        S = subs_full[:T]
    else:
        T = T_env
        E = env_full
        S = subs_full

    X_max = int(max(X_env, X_sub))
    # resample to a common width (max of the two)
    E_r = _resample_width(E, X_max)
    S_r = _resample_width(S, X_max)

    # normalize each to [0,1] to keep contrast
    En = _normalize_z(E_r)
    Sn = _normalize_z(S_r)

    fig = _fig_dark()
    # Env layer
    fig.add_trace(go.Heatmap(
        z=En,
        colorscale="Viridis",
        zsmooth=False,
        opacity=0.60,
        showscale=True,
        colorbar=dict(title="Env", x=1.02)
    ))
    # Substrate layer
    fig.add_trace(go.Heatmap(
        z=Sn,
        colorscale="Inferno",
        zsmooth=False,
        opacity=0.55,
        showscale=True,
        colorbar=dict(title="Substrate", x=1.10)
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=f"x (space, width={X_max})"),
        yaxis=dict(title="t (frames)", autorange="reversed"),  # t=0 at top looks like a raster over time
    )

    # one interactive, zoomable view
    placeholder.plotly_chart(fig, use_container_width=True)