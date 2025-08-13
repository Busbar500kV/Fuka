# core/physics.py
import numpy as np

def resample_row(row: np.ndarray, new_len: int) -> np.ndarray:
    """Periodic down/up-sample by integer mapping (keeps wrap)."""
    old_len = row.shape[0]
    if old_len == new_len:
        return row
    idx = (np.arange(new_len) * old_len // new_len) % old_len
    return row[idx]

def step_physics(prev_S: np.ndarray,
                 env_row: np.ndarray,
                 k_flux: float,
                 k_motor: float,
                 diffuse: float,
                 decay: float,
                 rng: np.random.Generator,
                 band: int = 3) -> tuple[np.ndarray, float]:
    """One local update step; returns (S_cur, flux_sum)."""
    # diffusion (nearest neighbors on ring)
    left  = np.roll(prev_S,  1)
    right = np.roll(prev_S, -1)
    S_diffused = prev_S + diffuse * (0.5*(left + right) - prev_S)
    S_decayed  = (1.0 - decay) * S_diffused

    # boundary band pump
    grad = np.maximum(env_row[:band] - S_decayed[:band], 0.0)
    pump = np.zeros_like(S_decayed)
    pump[:band] += k_flux * grad

    # motor exploration noise (boundary band only)
    mot = np.zeros_like(S_decayed)
    mot[:band] += k_motor * rng.random(band)

    S_cur = S_decayed + pump + mot
    flux_sum = float(np.sum(pump))
    return S_cur, flux_sum