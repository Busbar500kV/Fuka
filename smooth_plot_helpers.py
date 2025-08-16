# --- smooth_plot_helpers.py (you can paste into your app.py if you prefer) ---
import numpy as np
import plotly.graph_objects as go

def _normalize(a):
    a = np.asarray(a, dtype=float)
    mn = np.nanmin(a); mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def ensure_combo_fig(ss_key: str, T: int, X_env: int, X_space: int, height: int = 480):
    import streamlit as st
    if ss_key not in st.session_state:
        fig = go.Figure()

        # Two heatmaps side-by-side on one y (time) axis:
        # Env on [0, X_env-1], Substrate on [X_env+gap, X_env+gap+X_space-1]
        gap = 4
        x_env   = np.arange(X_env)
        x_space = np.arange(X_space) + (X_env + gap)
        y = np.arange(T)

        En0 = np.zeros((T, X_env))
        Sn0 = np.zeros((T, X_space))

        fig.add_trace(go.Heatmap(
            z=En0, x=x_env, y=y, colorscale="Viridis",
            zmin=0.0, zmax=1.0, colorbar=dict(title="Env"),
            showscale=True, name="Env", zsmooth=False
        ))
        fig.add_trace(go.Heatmap(
            z=Sn0, x=x_space, y=y, colorscale="Inferno",
            zmin=0.0, zmax=1.0, colorbar=dict(title="Substrate"),
            showscale=True, name="Substrate", zsmooth=False
        ))

        fig.update_layout(
            height=height,
            xaxis_title="space (Env  |  Substrate)",
            yaxis_title="time",
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            font=dict(color="#E0E6F1"),
            # KEY: keep user zoom/pan between updates
            uirevision="stick"
        )

        st.session_state[ss_key] = fig
    return st.session_state[ss_key]

def update_combo_fig(fig, E_full, S_full):
    # Normalize both to [0,1] independently so their contrast is stable.
    En = _normalize(E_full)
    Sn = _normalize(S_full)

    # Update heatmap z data only (no new figure)
    fig.data[0].z = En
    fig.data[1].z = Sn
    return fig

def draw_energy_timeseries_live(ph, t, e_cell, e_env, e_flux, uirev="stick"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env",  mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux", mode="lines"))
    fig.update_layout(
        height=240, uirevision=uirev,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#E0E6F1")
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None)