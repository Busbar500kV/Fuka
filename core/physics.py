# core/physics.py
import numpy as np

def diffuse_decay(prev: np.ndarray, diffuse: float, decay: float) -> np.ndarray:
    left = np.roll(prev, 1)
    right = np.roll(prev, -1)
    d = prev + diffuse * (0.5 * (left + right) - prev)
    return (1.0 - decay) * d

def pump(e_row: np.ndarray, s_row: np.ndarray, band: int, k_flux: float) -> np.ndarray:
    band = max(1, int(band))
    grad = np.maximum(e_row[:band] - s_row[:band], 0.0)
    out = np.zeros_like(s_row)
    out[:band] += k_flux * grad
    return out

def motor(band: int, k_motor: float, rng) -> np.ndarray:
    band = max(1, int(band))
    return k_motor * rng.random(band) if k_motor > 0 else np.zeros(band, dtype=float)

def add_noise(shape, k_noise, rng):
    return (k_noise * rng.normal(0.0, 1.0, size=shape)) if k_noise > 0 else 0.0