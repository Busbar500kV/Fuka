# core/engine.py
import numpy as np
from .config import Config
from .env import build_env
from .organism import History
from .physics import step_physics, resample_row

from core.connections import init_connections, grow_connections, prune_connections, plasticity_step, gaussian_kernels
self.conn = init_connections(X=cfg.space, N=cfg.n_conn, rng=self.rng,
                             n_seed=cfg.n_seed,
                             ell_prior=(cfg.ell_min, cfg.ell_max),
                             tau_prior=(cfg.tau_min, cfg.tau_max),
                             energy0=0.0)

class Engine:
    """Streaming-capable engine with simple local rules."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Build full env timeline E(t,x_env)
        self.env = build_env(cfg.env, self.rng)            # shape (T, X_env)
        self.T   = cfg.frames
        self.X   = cfg.space

        # Substrate S(t,x) (store whole timeline for full heatmap; still cheap)
        self.S   = np.zeros((self.T, self.X), dtype=float)

        # History
        self.hist = History()

    def step(self, t: int):
        e_row = self.env[t]
        # resample env row to substrate resolution
        e_row = resample_row(e_row, self.X)

        prev = self.S[t-1] if t > 0 else self.S[0]
        cur, flux = step_physics(
            prev_S=prev,
            env_row=e_row,
            k_flux=self.cfg.k_flux,
            k_motor=self.cfg.k_motor,
            diffuse=self.cfg.diffuse,
            decay=self.cfg.decay,
            rng=self.rng,
            band=3,
        )
        self.S[t] = cur

        # Bookkeep
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(e_row)))
        self.hist.E_flux.append(float(flux))

    def run(self, progress_cb=None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)