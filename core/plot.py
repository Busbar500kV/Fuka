# core/plots.py
import numpy as np
import plotly.graph_objects as go
from .organism import History

def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    rng = a.max() - a.min()
    if rng <= 1e-12:
        return np.zeros_like(a)
    return (a - a.min()) / rng

def draw_energy_timeseries(placeholder, hist: History):
    t  = hist.t if len(hist.t) else [0]
    Ec = hist.E_cell if len(hist.E_cell) else [0.0]
    Ee = hist.E_env  if len(hist.E_env)  else [0.0]
    Ef = hist.E_flux if len(hist.E_flux) else [0.0]

    fig = go.Figure()
    fig.add_scatter(x=t, y=Ec, mode="lines", name="E_cell")
    fig.add_scatter(x=t, y=Ee, mode="lines", name="E_env")
    fig.add_scatter(x=t, y=Ef, mode="lines", name="E_flux")
    fig.update_layout(
        height=280, margin=dict(l=20,r=10,t=30,b=30),
        template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    placeholder.plotly_chart(fig, use_container_width=True)

def draw_overlay_last_frame(placeholder, E: np.ndarray, S: np.ndarray):
    """Overlay env and substrate at the current (last) frame with zoom/pan."""
    if S.shape[0] == 0:
        placeholder.empty()
        return
    t = S.shape[0] - 1
    # resample E[t] to substrate width if needed
    e_row = E[t]
    xS = S.shape[1]
    if e_row.shape[0] != xS:
        idx = (np.arange(xS) * e_row.shape[0] // xS) % e_row.shape[0]
        e_row = e_row[idx]
    s_row = S[t]

    En = _normalize(e_row)
    Sn = _normalize(s_row)

    x = np.arange(xS)
    fig = go.Figure()
    fig.add_scatter(x=x, y=En, mode="lines", name="Env (last frame)")
    fig.add_scatter(x=x, y=Sn, mode="lines", name="Substrate (last frame)")
    fig.update_layout(
        height=280, margin=dict(l=20,r=10,t=30,b=30),
        template="plotly_dark",
        xaxis_title="x", yaxis_title="normalized amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    placeholder.plotly_chart(fig, use_container_width=True)

def draw_heatmap_full(placeholder, A: np.ndarray, title: str = "Heatmap"):
    """Show the full T×X array as a single heatmap (dark theme)."""
    # Normalize per-plot for contrast
    An = _normalize(A)
    fig = go.Figure(data=go.Heatmap(
        z=An, colorscale="Viridis", colorbar=dict(title="norm")
    ))
    fig.update_layout(
        title=title,
        height=320, margin=dict(l=20,r=10,t=35,b=30),
        template="plotly_dark",
        xaxis_title="x (space)", yaxis_title="t (frames)",
        yaxis=dict(autorange="reversed")  # show t=0 at top
    )
    placeholder.plotly_chart(fig, use_container_width=True)