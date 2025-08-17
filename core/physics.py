# core/physics.py
import numpy as np
from scipy.ndimage import gaussian_filter1d


def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1D environment row to match substrate resolution.
    Uses simple linear interpolation.
    """
    src_len = len(row)
    if src_len == target_len:
        return row
    x_src = np.linspace(0, 1, src_len)
    x_tgt = np.linspace(0, 1, target_len)
    return np.interp(x_tgt, x_src, row)


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
) -> tuple[np.ndarray, float]:
    """
    One physics update step.

    Args:
        prev_S: previous substrate (1D array)
        env_row: environment drive (1D array, same length after resample)
        k_flux: coupling from environment to substrate
        k_motor: self-driven activity
        diffuse: gaussian smoothing coefficient
        decay: decay factor per step
        rng: numpy RNG
        band: gaussian smoothing bandwidth
        conn_mask: optional per-cell modulation (same length as substrate).
                   If provided, multiplies the env_row before flux.

    Returns:
        (new substrate row, net flux value)
    """
    X = len(prev_S)

    # External flux (environment input)
    drive = env_row.copy()
    if conn_mask is not None:
        drive *= conn_mask

    flux = k_flux * (drive - prev_S)

    # Motor/self feedback
    motor = k_motor * prev_S

    # Update
    new_S = prev_S + flux + motor

    # Diffusion (cheap gaussian blur for local spreading)
    if diffuse > 0:
        new_S = gaussian_filter1d(new_S, sigma=band) * (1.0 - diffuse) + new_S * diffuse

    # Decay
    new_S *= (1.0 - decay)

    # Small noise
    new_S += 0.01 * rng.standard_normal(size=X)

    return new_S, float(np.mean(np.abs(flux)))