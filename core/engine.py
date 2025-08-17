# core/engine.py
# Streaming-capable engine with simple local rules.
# - Keeps the original physics & history behavior.
# - Optionally wires in "connections" if core.connections is available.
#   If not available, the simulation still runs.

from __future__ import annotations
import numpy as np

from .config import Config
from .env import build_env
from .organism import History
from .physics import step_physics, resample_row

# --- Optional connections layer ---------------------------------------------
_HAS_CONN = False
try:
    from .connections import (
        init_connections,
        grow_connections,
        prune_connections,
        plasticity_step,
        gaussian_kernels,
    )
    _HAS_CONN = True
except Exception:
    # Run without connections if the module isn't present or has errors.
    _HAS_CONN = False


class Engine:
    """Streaming-capable engine with simple local rules and optional connections."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Build full env timeline E(t, x_env)
        self.env = build_env(cfg.env, self.rng)           # shape (T_env, X_env)
        self.T   = int(cfg.frames)
        self.X   = int(cfg.space)

        # Substrate S(t, x) – we keep whole timeline (cheap, helps heatmaps)
        self.S   = np.zeros((self.T, self.X), dtype=float)

        # History (scalars per step)
        self.hist = History()

        # Optional connections
        self.conn = None
        self.kernels = None
        if _HAS_CONN:
            # Read connection-related knobs if present in cfg; otherwise use sensible defaults
            n_conn           = getattr(cfg, "n_conn", 64)
            n_seed           = getattr(cfg, "n_seed", 8)
            ell_min          = getattr(cfg, "ell_min", 1.5)
            ell_max          = getattr(cfg, "ell_max", 24.0)
            tau_min          = getattr(cfg, "tau_min", 0.0)
            tau_max          = getattr(cfg, "tau_max", 0.0)
            energy0          = getattr(cfg, "energy0", 0.0)

            # Growth/prune cadence
            self.conn_grow_every   = int(getattr(cfg, "conn_grow_every", 200))
            self.conn_grow_budget  = int(getattr(cfg, "conn_grow_budget", 4))
            self.conn_prune_every  = int(getattr(cfg, "conn_prune_every", 400))
            self.conn_prune_thresh = float(getattr(cfg, "conn_prune_thresh", -1e9))  # unused if your prune fn ignores

            # Initialize connection pool
            self.conn = init_connections(
                X=self.X,
                N=n_conn,
                rng=self.rng,
                n_seed=n_seed,
                ell_prior=(ell_min, ell_max),
                tau_prior=(tau_min, tau_max),
                energy0=energy0,
            )

            # Precompute gaussian kernels for quick inspection/plotting
            # (Most UIs just show a single “gate kernel” plot; here we keep the whole bank handy.)
            self.kernels = gaussian_kernels(self.conn, self.X)

        # Convenience aliases for UI
        self.env_full = self.env
        self.S_full   = self.S

    # -------------------------------------------------------------------------
    def step(self, t: int):
        """Advance one time step."""
        # Environment row (resample to substrate resolution if needed)
        e_row = self.env[t % self.env.shape[0]]
        e_row = resample_row(e_row, self.X)

        # Previous substrate
        prev = self.S[t-1] if t > 0 else self.S[0]

        # Core physics update
        cur, flux = step_physics(
            prev_S=prev,
            env_row=e_row,
            k_flux=float(self.cfg.k_flux),
            k_motor=float(self.cfg.k_motor),
            diffuse=float(self.cfg.diffuse),
            decay=float(self.cfg.decay),
            rng=self.rng,
            band=3,
        )
        self.S[t] = cur

        # History bookkeeping
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(e_row)))
        self.hist.E_flux.append(float(flux))

        # Optional: connections learning & lifecycle
        if _HAS_CONN and (self.conn is not None):
            # Local plasticity (update per-connection energy/params) using current signals
            # Your plasticity_step can be as simple or complex as you like. We pass
            # the local env row and the current substrate snapshot.
            plasticity_step(self.conn, env_row=e_row, subs_row=cur, rng=self.rng)

            # Periodic growth
            if self.conn_grow_every > 0 and (t % self.conn_grow_every == 0) and (t > 0):
                grow_connections(
                    self.conn,
                    X=self.X,
                    rng=self.rng,
                    budget=self.conn_grow_budget,
                )
                # If your gaussian_kernels depend on (ell, tau) and count, refresh the bank
                self.kernels = gaussian_kernels(self.conn, self.X)

            # Periodic pruning
            if self.conn_prune_every > 0 and (t % self.conn_prune_every == 0) and (t > 0):
                prune_connections(self.conn, thresh=self.conn_prune_thresh)
                self.kernels = gaussian_kernels(self.conn, self.X)

    # -------------------------------------------------------------------------
    def run(self, progress_cb=None):
        """Run the whole timeline; optionally stream progress via callback(t)."""
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)