# sim_core.py
# Fuka – minimal but stable engine with clear API + diagnostics
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional
import numpy as np

__all__ = [
    "Config",
    "default_config",
    "Engine",
    "make_engine",
]

# --------------------
# Configuration object
# --------------------
@dataclass
class Config:
    frames: int = 1600
    space: int = 192
    seed: int = 0

    # environment & energy
    env_power: float = 1.0        # baseline free energy in the environment
    env_hotspots: int = 6         # moving “energy rays”
    env_decay: float = 0.985      # how quickly environment diffuses per step
    pool_init: float = 50.0       # initial intracellular energy

    # connections (kept simple—role emerges from drive mix)
    n_conns: int = 12
    act_cost: float = 0.02        # cost paid when a conn activates
    main_cost: float = 0.001      # base upkeep

    # drives (relative weights used to turn activation into harvest)
    w_direct: float = 1.0         # “sensor-like” drive from environment mismatch
    w_indirect: float = 0.6       # “internal-like” drive from substrate mismatch
    w_motor: float = 0.8          # “motor-like” drive that changes environment

    # stability guards
    clip_energy: float = 1e6
    dt: float = 1.0               # one tick per frame

def default_config() -> Dict:
    """Return a plain dict so Streamlit widgets can mutate it easily."""
    return asdict(Config())


# -------------
# Core engine
# -------------
class Engine:
    """
    Extremely compact simulation:
      - 1D space with an environment field E[t, x] carrying free energy.
      - A substrate S[t, x] inside the ‘cell’ that can absorb/emit energy.
      - A small set of generic connections whose activation is driven by
        local mismatches; their ‘role’ (sensor/internal/motor) is emergent.
    """

    def __init__(self, cfg: Dict):
        self.cfg = Config(**cfg) if isinstance(cfg, dict) else cfg
        self.rng = np.random.default_rng(self.cfg.seed)

        T = self.cfg.frames
        X = self.cfg.space
        C = self.cfg.n_conns

        # State history buffers
        self.energy = np.full(T, np.nan, dtype=np.float64)
        self.env = np.zeros((T, X), dtype=np.float64)
        self.subs = np.zeros((T, X), dtype=np.float64)
        self.act = np.zeros((T, C), dtype=np.float64)  # activation per connection

        # Init
        self.energy[0] = float(self.cfg.pool_init)
        self._init_environment(self.env[0])
        self.subs[0] = 0.0

        # Per‑connection location & phase (space-time “event” style)
        self.c_pos = self.rng.integers(0, X, size=C)
        self.c_phase = self.rng.random(C) * 2.0 * np.pi
        self.c_speed = self.rng.uniform(0.2, 1.0, size=C)  # how fast they “sweep”

        # Simple kernels so nothing explodes
        self._env_kernel = np.exp(-((np.arange(-6, 7)) ** 2) / 12.0)
        self._env_kernel /= self._env_kernel.sum()

        self._t = 0  # current frame

    # -- Environment set up
    def _init_environment(self, env_x: np.ndarray) -> None:
        X = env_x.shape[0]
        env_x[:] = 0.0
        # place a few initial hotspots
        idx = self.rng.choice(X, size=max(1, self.cfg.env_hotspots // 2), replace=False)
        env_x[idx] = self.cfg.env_power * 4.0

    def _advance_environment(self, t: int) -> None:
        """Create smooth moving ‘rays’ of free energy, diffusing a bit."""
        E_prev = self.env[t - 1]
        X = E_prev.shape[0]

        # diffuse/decay
        E = self.cfg.env_decay * E_prev.copy()

        # inject moving rays at deterministic angles (no trig used)
        # We emulate diagonal motion by cyclically shifting short pulses.
        n_rays = self.cfg.env_hotspots
        stride = max(3, X // (n_rays + 1))
        for r in range(n_rays):
            base = (r * stride + (t * (r + 1))) % X
            E[base] += self.cfg.env_power * (1.0 + 0.2 * ((r + t) % 3))

        # smooth via tiny convolution to keep things sane
        E = np.convolve(
            np.pad(E, (6, 6), mode="wrap"), self._env_kernel, mode="valid"
        )

        self.env[t] = E

    # -- One simulation step
    def _step(self, t: int) -> None:
        cfg = self.cfg
        X = cfg.space
        C = cfg.n_conns

        # 1) advance environment
        self._advance_environment(t)

        # 2) compute local mismatches
        env_t = self.env[t]
        subs_prev = self.subs[t - 1]
        # mismatch w.r.t. zero for substrate (want to equalize), and env drives
        d_env = env_t - subs_prev
        d_sub = -(subs_prev - subs_prev.mean())

        # 3) connection activation (purely local; 0..1)
        # each connection samples a small window around its position
        a = np.zeros(C, dtype=np.float64)
        for k in range(C):
            # event-like wandering of connection along space
            self.c_pos[k] = (self.c_pos[k] + int(self.c_speed[k])) % X

            i = self.c_pos[k]
            j0 = (i - 2) % X
            j1 = (i + 3) % X
            if j0 < j1:
                local_env = d_env[j0:j1]
                local_sub = d_sub[j0:j1]
            else:
                local_env = np.concatenate([d_env[j0:], d_env[:j1]])
                local_sub = np.concatenate([d_sub[j0:], d_sub[:j1]])

            # bounded, simple “softplus-like” without overflow
            def softplus_bounded(x: np.ndarray) -> float:
                x = np.clip(x, -20.0, 20.0)
                return float(np.log1p(np.exp(x)).mean())

            drive = (
                cfg.w_direct * softplus_bounded(local_env)
                + cfg.w_indirect * softplus_bounded(local_sub)
            )
            # slight “motor” bias to push substrate towards env where active
            motor = cfg.w_motor * softplus_bounded(local_env - local_sub)

            # convert to bounded activation
            a[k] = np.tanh(0.2 * (drive + 0.5 * motor))

        # 4) harvest vs cost
        # harvest from better env/subs match; costs for activity and upkeep
        harvest = 0.02 * a.sum() * float(np.maximum(d_env.var(), 1e-6))
        cost = cfg.main_cost * C + cfg.act_cost * (a > 0.05).sum()

        # 5) update substrate towards environment using activations as pumps
        subs_t = subs_prev.copy()
        # local pumps “pull” towards env where active
        for k in range(C):
            i = self.c_pos[k]
            j0 = (i - 1) % X
            j1 = (i + 2) % X
            if j0 < j1:
                region = slice(j0, j1)
                subs_t[region] += 0.05 * a[k] * (env_t[region] - subs_t[region])
            else:
                idx = np.r_[np.arange(j0, X), np.arange(0, j1)]
                subs_t[idx] += 0.05 * a[k] * (env_t[idx] - subs_t[idx])

        # write state
        self.subs[t] = subs_t
        self.act[t] = a

        # 6) energy pool
        self.energy[t] = np.clip(
            self.energy[t - 1] + harvest - cost, -cfg.clip_energy, cfg.clip_energy
        )

    # -- Public API
    def run(self, progress_cb: Optional[Callable[[int], None]] = None) -> None:
        T = self.cfg.frames
        for t in range(1, T):
            self._step(t)
            self._t = t
            if progress_cb and (t % 10 == 0 or t == T - 1):
                progress_cb(t)

    def snapshots(self) -> Dict[str, np.ndarray]:
        return dict(
            energy=self.energy.copy(),
            env=self.env.copy(),
            subs=self.subs.copy(),
            act=self.act.copy(),
        )


def make_engine(cfg: Dict) -> Engine:
    """Factory used by the Streamlit app."""
    return Engine(cfg)