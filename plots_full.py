# plots_full.py
# Full-history space–time plots for Environment E(t,x) and Substrate S(t,x)
# - Black theme
# - Zoom/Pan enabled
# - One live-updating stacked figure (two heatmaps) or an RGB overlay option

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Helpers
# ----------------------------

def _resample_to_width(arr: np.ndarray, new_x: int) -> np.ndarray:
    """
    Resample array (T, X_old) to (T, new_x) by nearest indexing.
    Keeps periodic structure if you used a ring, but is cheap and stable.
    """
    if arr.shape[1] == new_x:
        return arr
    x_old = arr.shape[1]
    idx = (np.arange(new_x) * x_old // new_x) % x_old
    return arr[:, idx]

def _normalize01(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    zmin = float(np.nanmin(z))
    zptp = float(np.nanmax(z) - zmin)
    return (z - zmin) / (zptp + eps)

def _dark_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="#EAEAEA"),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig

# ----------------------------
# Public: stacked heatmaps
# ----------------------------

def fig_full_stacked(
    E_full: np.ndarray,     # shape (T, X_env)
    S_full: np.ndarray,     # shape (T, X_space)
    x_env: Optional[int] = None,  # if provided, resample both to this width for exact alignment
    title: str = "Environment E(t,x) • Substrate S(t,x)",
    env_cmap: str = "Viridis",
    subs_cmap: str = "Magma",
) -> go.Figure:
    """
    Returns a single figure with two heatmaps (ENV, SUBS) stacked.
    X-axis shows the **full spatial width** (env length),
    Y-axis shows the **full timeline** (all frames).
    """
    E = np.asarray(E_full)
    S = np.asarray(S_full)

    # Decide a common width for nicer zoom/compare
    width = x_env if x_env is not None else E.shape[1]
    E2 = _resample_to_width(E, width)
    S2 = _resample_to_width(S, width)

    T = E2.shape[0]
    x = np.arange(width)
    t = np.arange(T)

    fig = make_subplots(
        rows=2, cols=1, shared_x=True,
        subplot_titles=("Environment E(t,x)", "Substrate S(t,x)")
    )

    fig.add_trace(
        go.Heatmap(
            z=E2, x=x, y=t,
            colorscale=env_cmap, colorbar=dict(title="E"),
            zsmooth=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=S2, x=x, y=t,
            colorscale=subs_cmap, colorbar=dict(title="S"),
            zsmooth=False
        ),
        row=2, col=1
    )

    # Axes: full extents; time downward (like a raster) is often nice -> reverse y
    fig.update_xaxes(title_text="space (x)", row=2, col=1, range=[0, width-1])
    fig.update_yaxes(title_text="frame (t)", row=1, col=1, range=[T-1, 0])
    fig.update_yaxes(title_text="frame (t)", row=2, col=1, range=[T-1, 0])

    _dark_layout(fig, title)
    return fig

# ----------------------------
# Public: RGB overlay heatmap (optional)
# ----------------------------

def fig_full_overlay(
    E_full: np.ndarray,
    S_full: np.ndarray,
    x_env: Optional[int] = None,
    title: str = "Combined (E in G, S in R)",
    r_gain: float = 1.0,
    g_gain: float = 1.0,
    b_gain: float = 0.0,   # keep 0.0 for two‑channel overlay; >0 to add blue = |E-S|
) -> go.Figure:
    """
    Builds an RGB image where:
      R channel = normalized Substrate
      G channel = normalized Environment
      B channel = optional |E - S| contrast
    This gives a single zoomable heatmap with both quantities.
    """
    E = np.asarray(E_full)
    S = np.asarray(S_full)
    width = x_env if x_env is not None else E.shape[1]
    E2 = _resample_to_width(E, width)
    S2 = _resample_to_width(S, width)

    T = E2.shape[0]

    En = _normalize01(E2)
    Sn = _normalize01(S2)
    Bn = _normalize01(np.abs(E2 - S2)) if b_gain > 0 else np.zeros_like(En)

    rgb = np.clip(
        np.stack([r_gain * Sn, g_gain * En, b_gain * Bn], axis=-1), 0.0, 1.0
    )

    # Plotly doesn't take RGB arrays directly in Heatmap; we convert to hex colors
    # shape (T, X) where each entry is a hex color string.
    rgb8 = (rgb * 255).astype(np.uint8)
    hex_img = np.apply_along_axis(
        lambda px: f"#{px[0]:02x}{px[1]:02x}{px[2]:02x}", 2, rgb8
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=np.zeros((T, width)),  # dummy z so we get the heatmap grid
            x=np.arange(width),
            y=np.arange(T),
            colorscale=[[0, "#000000"], [1, "#ffffff"]],  # will be overridden by zmid trick
            showscale=False,
            zsmooth=False,
            hoverinfo="skip",
        )
    )
    # Replace cell colors via image; use add_layout_image for speed
    fig.add_layout_image(
        dict(
            source=_rgb_to_pil_image(rgb8),
            xref="x", yref="y",
            x=0, y=0,
            sizex=width, sizey=T,
            sizing="stretch",
            layer="below"
        )
    )

    # Axes
    fig.update_xaxes(title_text="space (x)", range=[0, width-1])
    fig.update_yaxes(title_text="frame (t)", range=[T-1, 0])  # time downward

    _dark_layout(fig, title)
    return fig

def _rgb_to_pil_image(rgb8: np.ndarray):
    """Small helper to turn (T, X, 3) uint8 RGB into a PIL image for Plotly layout_image."""
    from PIL import Image
    # Plotly expects (width, height) mapping; PIL uses (W,H). Our array is (T,H)=(rows,cols)
    # We need to flip y so that y increases downward.
    img = Image.fromarray(np.flipud(rgb8), mode="RGB")
    return img

# ----------------------------
# Streamlit integration
# ----------------------------

def streamlit_live_stacked(st, placeholder, get_arrays_fn, width: Optional[int] = None, title: str = "Environment • Substrate"):
    """
    Live updater for stacked heatmaps.
    - `st`: the streamlit module
    - `placeholder`: a st.empty() container
    - `get_arrays_fn`: callable -> (E_full, S_full) current arrays
    - `width`: optional resample width to align X
    """
    E, S = get_arrays_fn()
    fig = fig_full_stacked(E, S, x_env=width, title=title)
    placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

def streamlit_live_overlay(st, placeholder, get_arrays_fn, width: Optional[int] = None, title: str = "Combined (E in G, S in R)"):
    """
    Live updater for overlay heatmap.
    """
    E, S = get_arrays_fn()
    fig = fig_full_overlay(E, S, x_env=width, title=title)
    placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})