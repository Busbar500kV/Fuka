# core/engine.py
import numpy as np
from typing import Optional, Callable, Tuple

from .config import Config
from .history import History
from .env import build_env
from .physics import diffuse_decay, pump, motor, add_noise
from .organism import advance_boundary
from .connections import GateKernel

class Engine:
    """
    Streaming engine: maintains env (T_env, X_env), substrate S (T, X),
    a moving boundary offset, and a learnable gate kernel.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = build_env(cfg.env, self.rng)  # (T_env, X_env)
        self.T = int(cfg.frames)
        self.X = int(cfg.space)
        self.S = np.zeros((self.T, self.X), dtype=float)
        self.hist = History()
        self.offset = 0.0
        self.kernel = GateKernel(cfg.kernel)

    def _sample_env_row(self, t: int) -> np.ndarray:
        E = self.env
        X_env = E.shape[1]
        base = (np.arange(self.X) * X_env // self.X)  # periodic resample
        idx = (base + int(self.offset)) % X_env
        ti = min(t, E.shape[0] - 1)
        return E[ti, idx]

    # expose for UI
    def _row(self, t: int) -> np.ndarray:
        return self._sample_env_row(t)

    def step(self, t: int) -> Tuple[float, float, float]:
        c = self.cfg
        e_row = self._sample_env_row(t)

        prev = self.S[t - 1] if t > 0 else self.S[0]
        s0 = diffuse_decay(prev, c.diffuse, c.decay)

        # pump + motor in boundary band
        pu = pump(e_row, s0, c.band, c.k_flux)
        mo = np.zeros_like(s0)
        mo[:int(c.band)] = motor(c.band, c.k_motor, self.rng)

        # gate kernel (general connection)
        gated = np.zeros_like(s0)
        if c.kernel.enabled and len(self.kernel.w) > 0 and c.kernel.k_gate != 0.0:
            gated = c.kernel.k_gate * self.kernel.convolve(s0)

        noise = add_noise(s0.shape, c.k_noise, self.rng)
        cur = s0 + pu + mo + gated + noise
        if c.cap > 0:
            np.clip(cur, 0.0, float(c.cap), out=cur)
        self.S[t] = cur

        # Local plasticity for kernel (uses early window for stability)
        if c.kernel.enabled and c.kernel.lr > 0.0:
            self.kernel.plasticity(s0, e_row, cur)

        motor_mass = float(np.sum(mo[:int(c.band)]))
        self.offset = advance_boundary(self.offset, motor_mass, c.boundary_speed, self.env.shape[1])

        # bookkeeping
        E_cell = float(np.mean(cur))
        E_env = float(np.mean(e_row))
        E_flux = float(np.sum(pu))
        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)
        return E_cell, E_env, E_flux

    def run(self, progress_cb: Optional[Callable[[int], None]] = None, snapshot_every: Optional[int] = None):
        # snapshot_every is accepted for API compatibility; the UI handles chunking.
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)