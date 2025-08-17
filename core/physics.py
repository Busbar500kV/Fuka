# core/physics.py
from __future__ import annotations
import numpy as np

def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1D environment row to match substrate resolution via linear interp.
    """
    src_len = int(row.shape[0])
    if src_len == target_len:
        return row
    x_src = np.linspace(0.0, 1.0, src_len)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, row).astype(np.float64)


def step_physics(
    prev_S: np.ndarray,
    env_row: np.ndarray,
    k_flux: float,
    k_motor: float,
    diffuse: float,
    decay: float,
    rng: np.random.Generator,
    band: int = 3,
    conn_mask: np.ndarray | None = None,
    *,
    c_motor: float = 0.02,         # <- motor energy cost (prevents runaway)
    noise_sigma: float = 0.0,      # <- optional small noise
    vmax: float = 10.0             # <- soft clip to keep S bounded
) -> tuple[np.ndarray, float]:
    """
    One physics update step with bounded dynamics.

    Args:
        prev_S   : previous substrate (1D array)
        env_row  : environment drive (1D array, same length as prev_S after resample)
        k_flux   : coupling from environment to substrate, pulls S toward env
        k_motor  : exploratory push at the boundary (or where conn_mask > 0)
        diffuse  : diffusion strength (0..~0.5). Uses discrete Laplacian.
        decay    : fractional decay per step (0..1)
        rng      : numpy Generator
        band     : if conn_mask is None, a [0:band] region acts as “boundary”
        conn_mask: optional per-cell modulation (same length). Multiplies env before flux.

    Returns:
        new_S, pos_flux_sum
        - new_S: new substrate row
        - pos_flux_sum: sum of *positive* flux (incoming) this step
    """
    X = int(prev_S.shape[0])

    # 1) External flux (environment-to-S)
    drive = env_row.astype(np.float64).copy()
    if conn_mask is not None:
        # mask the coupling — “gates” where env can couple in
        drive *= conn_mask
    flux = k_flux * (drive - prev_S)
    # We'll report only incoming (positive) flux as “harvested energy”
    pos_flux_sum = float(np.sum(np.maximum(flux, 0.0)))

    # 2) Motor: exploratory push only at boundary (or masked), with cost everywhere
    if conn_mask is not None:
        motor_mask = (conn_mask > 0).astype(np.float64)
    else:
        motor_mask = np.zeros(X, dtype=np.float64)
        motor_mask[:max(1, int(band))] = 1.0  # left boundary band

    # Random exploration localized to mask
    motor_push = k_motor * rng.standard_normal(X) * motor_mask

    # Motor cost: energy spent to push (scaled by current S)
    motor_cost = c_motor * prev_S

    # 3) Raw update before diffusion/decay
    new_S = prev_S + flux + motor_push - motor_cost

    # 4) Diffusion — discrete Laplacian (conservative local smoothing)
    if diffuse > 0.0:
        left  = np.roll(new_S,  1)
        right = np.roll(new_S, -1)
        lap = left + right - 2.0 * new_S
        new_S = new_S + diffuse * lap

    # 5) Decay — global leakage
    if decay > 0.0:
        new_S *= (1.0 - decay)

    # 6) Small zero-mean noise (optional)
    if noise_sigma > 0.0:
        new_S += noise_sigma * rng.standard_normal(X)

    # 7) Bound to physical range (nonnegative, capped)
    #    Use a soft cap by clipping; swap for np.tanh if you prefer soft saturation.
    new_S = np.clip(new_S, 0.0, vmax)

    return new_S, pos_flux_sum