# core/physics.py
import numpy as np

class Connection:
    """A 1D 'worldline' connection between two nodes in the substrate."""
    def __init__(self, a: int, b: int, energy: float = 0.0):
        self.a = a          # endpoint index (node index in env/substrate)
        self.b = b
        self.energy = energy
        self.age = 0

    def step(self,
             env: np.ndarray,
             k_absorb: float = 0.1,
             k_convert: float = 0.05,
             decay: float = 0.01,
             noise_amp: float = 0.0,
             rng: np.random.Generator | None = None) -> float:
        """One update step for this connection."""
        # sample free energy from environment at both ends
        e_in = 0.5 * (env[self.a] + env[self.b])
        e_abs = k_absorb * e_in

        # convert free → localized
        self.energy += k_convert * e_abs

        # decay (leakage back to free pool)
        self.energy *= (1.0 - decay)

        # random jitter for exploration
        if rng is not None and noise_amp > 0:
            self.energy += noise_amp * rng.standard_normal()

        # aging
        self.age += 1

        return self.energy


def step_physics(connections: list[Connection],
                 env: np.ndarray,
                 k_absorb: float = 0.1,
                 k_convert: float = 0.05,
                 decay: float = 0.01,
                 noise_amp: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Update all connections once; return array of energies."""
    return np.array([
        c.step(env, k_absorb, k_convert, decay, noise_amp, rng)
        for c in connections
    ])