# core/plot.py
# Pure plotting helpers for Streamlit + Plotly (dark theme).
# Backward compatible with older app.py calls.

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Iterable, Tuple, Optional


__all__ = [
    "draw_energy_timeseries",
    "draw_overlay_last_frame",
    "draw_heatmap_full",
]


# ---------- small utils ----------

def _fig_dark(title: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e5e5e5"),
        margin=dict(l=40, r=20, t=50, b=40),
        title=title or "",
    )
    return fig


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=float)
    rng = np.ptp(x)
    if rng == 0 or not np.isfinite(rng):
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmin(x)) / (rng + 1e-12)


def _nearest_resample_1d(x: np.ndarray, new_len: int) -> np.ndarray:
    """Nearest-neighbor resample a 1D array to new_len."""
    x = np.asarray(x)
    if x.size == new_len:
        return x
    idx = np.clip(
        np.round(np.linspace(0, max(1, x.size - 1), new_len)).astype(int),
        0, x.size - 1
    )
    return x[idx]


def _nearest_resample_2d(A: np.ndarray, new_T: int, new_X: int) -> np.ndarray:
    """Nearest-neighbor resample a 2D [T, X] array to (new_T, new_X)."""
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Expected 2D array [T, X] for resampling.")
    T, X = A.shape
    if T == new_T and X == new_X:
        return A
    t_idx = np.clip(
        np.round(np.linspace(0, max(1, T - 1), new_T)).astype(int),
        0, T - 1
    )
    x_idx = np.clip(
        np.round(np.linspace(0, max(1, X - 1), new_X)).astype(int),
        0, X - 1
    )
    return A[t_idx][:, x_idx]


# ============================================================
# 1) Energy time‑series (backward compatible signature)
#    New: draw_energy_timeseries(placeholder, t, e_cell, e_env, e_flux)
#    Old: draw_energy_timeseries(placeholder, hist)
# ============================================================

def draw_energy_timeseries(placeholder, *args):
    """
    Plot E_cell, E_env, E_flux vs t on a single figure (dark mode).

    Backward‑compatible:
      - New style:
          draw_energy_timeseries(ph, t, e_cell, e_env, e_flux)
      - Old style:
          draw_energy_timeseries(ph, hist)  # where hist has .t, .E_cell, .E_env, .E_flux
    """
    if len(args) == 1:
        # old style: args[0] is a history-like object
        hist = args[0]
        t = getattr(hist, "t", [])
        e_cell = getattr(hist, "E_cell", [])
        e_env  = getattr(hist, "E_env", [])
        e_flux = getattr(hist, "E_flux", [])
    elif len(args) == 4:
        # new style
        t, e_cell, e_env, e_flux = args
    else:
        raise TypeError(
            "draw_energy_timeseries expects either (placeholder, hist) or "
            "(placeholder, t, e_cell, e_env, e_flux)"
        )

    # Convert to numpy for safety
    t      = np.asarray(t, dtype=float)
    e_cell = np.asarray(e_cell, dtype=float)
    e_env  = np.asarray(e_env, dtype=float)
    e_flux = np.asarray(e_flux, dtype=float)

    fig = _fig_dark("Energy vs time")
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux", mode="lines"))
    fig.update_xaxes(title="t (frame)")
    fig.update_yaxes(title="energy (arb)")
    placeholder.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2) Overlay for the *last frame* (1D curves)
#    draw_overlay_last_frame(placeholder, E, S, title="…", normalize=True)
#    Accepts either 1D arrays or full 2D [T, X] and will pick last row.
#    Resamples one to the other's length if needed.
# ============================================================

def draw_overlay_last_frame(
    placeholder,
    E: np.ndarray,
    S: np.ndarray,
    title: str = "Environment vs Substrate — last frame",
    normalize: bool = True,
):
    """
    Draw a 1D overlay of environment and substrate at the last time frame.

    Accepts:
      - E, S as 1D arrays (same length preferred).
      - E, S as 2D arrays [T, X]. We take the last row (t = T-1).

    If lengths differ, S is resampled to match E by nearest-neighbor.
    """
    E = np.asarray(E, dtype=float)
    S = np.asarray(S, dtype=float)

    # Pick last frame if 2D
    if E.ndim == 2:
        E = E[-1]
    if S.ndim == 2:
        S = S[-1]

    # Make lengths match
    if E.size != S.size:
        S = _nearest_resample_1d(S, E.size)

    x = np.arange(E.size, dtype=float)

    if normalize:
        En = _normalize(E)
        Sn = _normalize(S)
        y1, y2 = En, Sn
        ylab = "normalized amplitude"
    else:
        y1, y2 = E, S
        ylab = "amplitude"

    fig = _fig_dark(title)
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines", name="Env (last)"))
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", name="Substrate (last)"))
    fig.update_xaxes(title="space (x)")
    fig.update_yaxes(title=ylab)
    placeholder.plotly_chart(fig, use_container_width=True)


# ============================================================
# 3) Full heatmaps with zoom (combined or stacked)
#    draw_heatmap_full(placeholder, E_full, S_full, mode="overlay"/"stacked", title=…)
#    - overlay: both heatmaps over the same axes w/ transparency
#    - stacked: two rows, Env on top, Substrate bottom, shared x
# ============================================================

def draw_heatmap_full(
    placeholder,
    E_full: np.ndarray,
    S_full: np.ndarray,
    mode: str = "overlay",
    title: str = "E(t, x) • S(t, x)",
    env_colorscale: str = "Viridis",
    sub_colorscale: str = "Inferno",
    sub_opacity: float = 0.60,
):
    """
    Draw the full time–space fields as heatmaps with Plotly zooming.

    Parameters
    ----------
    E_full : np.ndarray [T_e, X_e]
    S_full : np.ndarray [T_s, X_s]
    mode   : "overlay" or "stacked"
    """
    E_full = np.asarray(E_full, dtype=float)
    S_full = np.asarray(S_full, dtype=float)

    if E_full.ndim != 2 or S_full.ndim != 2:
        raise ValueError("E_full and S_full must be 2D arrays [T, X].")

    Te, Xe = E_full.shape
    Ts, Xs = S_full.shape

    # Resample substrate to the environment grid (time + space) so both align
    S_rs = _nearest_resample_2d(S_full, new_T=Te, new_X=Xe)

    En = _normalize(E_full)
    Sn = _normalize(S_rs)

    if mode.lower() == "overlay":
        fig = _fig_dark(title)
        # Env base layer
        fig.add_trace(
            go.Heatmap(
                z=En,
                colorscale=env_colorscale,
                showscale=False,
                name="Env",
                hovertemplate="t=%{y}, x=%{x}, Env=%{z:.3f}<extra></extra>",
                zsmooth=False,
            )
        )
        # Substrate overlay
        fig.add_trace(
            go.Heatmap(
                z=Sn,
                colorscale=sub_colorscale,
                opacity=sub_opacity,
                showscale=False,
                name="Substrate",
                hovertemplate="t=%{y}, x=%{x}, Sub=%{z:.3f}<extra></extra>",
                zsmooth=False,
            )
        )
        fig.update_xaxes(title="space x")
        fig.update_yaxes(title="time t")

    elif mode.lower() == "stacked":
        # Create two y domains manually to stay in one figure (so zoom syncs)
        fig = _fig_dark(title)
        # Top (Env)
        fig.add_trace(
            go.Heatmap(
                z=En,
                colorscale=env_colorscale,
                showscale=False,
                name="Env",
                hovertemplate="t=%{y}, x=%{x}, Env=%{z:.3f}<extra></extra>",
                zsmooth=False,
            ),
            row=None, col=None
        )
        # Manually assign axes domains
        fig.update_yaxes(domain=[0.55, 1.0], title="time t (Env)", matches=None)
        fig.update_xaxes(domain=[0.0, 1.0], title="", matches=None)

        # Bottom (Substrate) — we emulate a second panel by adding a new trace+axis
        fig.add_trace(
            go.Heatmap(
                z=Sn,
                colorscale=sub_colorscale,
                showscale=False,
                name="Substrate",
                hovertemplate="t=%{y}, x=%{x}, Sub=%{z:.3f}<extra></extra>",
                zsmooth=False,
                yaxis="y2",
                xaxis="x2",
            )
        )
        fig.update_layout(
            xaxis2=dict(anchor="y2", domain=[0.0, 1.0], matches="x"),
            yaxis2=dict(anchor="x2", domain=[0.0, 0.45], title="time t (Substrate)"),
        )
        fig.update_xaxes(title="space x", row=None, col=None)

    else:
        raise ValueError("mode must be 'overlay' or 'stacked'")

    placeholder.plotly_chart(fig, use_container_width=True)