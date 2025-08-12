# sim_core.py
# Free‑energy gradient engine with general connections (kernels),
# local plasticity, growth/prune, and boundary motion.
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


# ----------------------------
# Environment (space–time field)
# ----------------------------

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 5000
    noise_sigma: float = 0.01
    # sources: list of dicts: {"kind":"moving_peak"/"static_peak",
    #                          "amp":float,"speed":float,"width":float,"start":int}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "static_peak",  "amp": 1.0, "speed": 0.0,  "width": 4.0, "start": 340},
    ])


def _peak_ring(x_count: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(x_count, dtype=float)
    d = np.minimum(np.abs(x - pos), x_count - np.abs(x - pos))
    return amp * np.exp(-(d * d) / (2.0 * max(width, 1e-6) ** 2))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)
    for s in cfg.sources:
        kind = s.get("kind", "moving_peak")
        amp = float(s.get("amp", 1.0))
        speed = float(s.get("speed", 0.0)) * X  # cells per frame
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X
        pos = float(start)
        for t in range(T):
            E[t] += _peak_ring(X, pos, amp, width)
            if kind == "moving_peak":
                pos = (pos + speed) % X
    if cfg.noise_sigma > 0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)
    np.maximum(E, 0.0, out=E)
    return E


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64                     # substrate height (cells)
    band: int = 2                       # active boundary width (cells)
    diffuse: float = 0.15
    decay: float = 0.01

    # energy / motion
    k_flux: float = 0.08
    k_motor: float = 1.0
    motor_noise: float = 0.02
    c_motor: float = 0.80
    alpha_move: float = 0.3
    beta_tension: float = 0.02

    # kernels (general connections)
    gate_win: int = 30                  # half window K
    eta: float = 0.02                   # learning rate
    ema_beta: float = 0.10              # baseline EMA
    lam_l1: float = 0.10
    prune_thresh: float = 0.10
    min_age: int = 200
    spawn_rate: float = 0.02
    max_conns: int = 64

    env: FieldCfg = field(default_factory=FieldCfg)


# ----------------------------
# History buffer
# ----------------------------

@dataclass
class History:
    t: List[int] = field(default_factory=list)
    E_cell: List[float] = field(default_factory=list)
    E_env: List[float] = field(default_factory=list)
    E_flux: List[float] = field(default_factory=list)


# ----------------------------
# Engine with kernels
# ----------------------------

class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # env & simple resampler to substrate width (space)
        self.env = build_env(cfg.env, self.rng)        # (T, X_env)
        self.T = cfg.frames
        self.X_env = self.env.shape[1]
        self.X = cfg.space

        # substrate field (store full film for heatmap)
        self.S = np.zeros((self.T, self.X), dtype=float)
        self.S[0] = 0.0

        # boundary location (float index in env ring)
        self.b = 0.0

        # general connections (kernels)
        self.K = cfg.gate_win
        self.w: List[np.ndarray] = []          # each shape (2K+1,)
        self.age: List[int] = []
        self.energy_accum: List[float] = []
        self.motion_accum: List[float] = []
        self.harv_accum: List[float] = []

        # baseline EMA for env slice
        self.ema_env = np.zeros(2 * self.K + 1, dtype=float)

        self.hist = History()

        # start with a couple random kernels
        for _ in range(min(4, cfg.max_conns)):
            self._spawn_kernel()

    # ----- helpers -----

    def _slice_env(self, t: int, center: float) -> np.ndarray:
        # take (2K+1) ring sample from env row t around float center in env-width coordinates
        idx0 = int(np.floor(center)) % self.X_env
        frac = center - np.floor(center)
        # linear interpolation between two ring points
        x = (np.arange(-self.K, self.K + 1) + idx0) % self.X_env
        x1 = (x + 1) % self.X_env
        row = self.env[t]
        base = row[x]
        nxt = row[x1]
        return (1.0 - frac) * base + frac * nxt

    def _spawn_kernel(self):
        K = self.K
        k = self.rng.normal(0.0, 0.05, size=(2 * K + 1,))
        self.w.append(k)
        self.age.append(0)
        self.energy_accum.append(0.0)
        self.motion_accum.append(0.0)
        self.harv_accum.append(0.0)

    # ----- one step -----

    def step(self, t: int) -> Tuple[float, float, float]:
        c = self.cfg

        # env -> substrate row resampled
        e_row_env = self.env[t]  # env width
        # resample env row to substrate width X for viewing / diffusion coupling
        idx = (np.arange(self.X) * self.X_env) // self.X
        e_row = e_row_env[idx]

        prev = self.S[t - 1] if t > 0 else self.S[0]
        left, right = np.roll(prev, 1), np.roll(prev, -1)
        s_diff = prev + c.diffuse * (0.5 * (left + right) - prev)
        s_dec = (1.0 - c.decay) * s_diff

        # boundary band (in substrate coordinates) touches x=0..band-1
        band = c.band
        grad = np.maximum(e_row[:band] - s_dec[:band], 0.0)
        pump = np.zeros_like(s_dec)
        pump[:band] += c.k_flux * grad

        # ----- kernels act on env slice around current boundary center -----
        slice_env = self._slice_env(t, self.b)
        self.ema_env = (1.0 - c.ema_beta) * self.ema_env + c.ema_beta * slice_env
        centered = slice_env - self.ema_env

        total_harv = 0.0
        total_motion = 0.0
        # cell energy baseline (EMA of mean S)
        E_cell_prev = float(np.mean(prev))
        # small “advantage” signal: will be recomputed after update below

        new_ws = []
        for i, wi in enumerate(self.w):
            # local drive from correlation with env centered slice
            drive = np.dot(wi, centered)
            drive = max(0.0, drive)

            # harvest: only if there is consumable gradient at band
            H = c.k_flux * drive * float(np.mean(grad))  # simple scalar coupling
            total_harv += H

            # motor from same drive
            M = c.k_motor * drive
            total_motion += M
            self.harv_accum[i] += H
            self.motion_accum[i] += M

            # L1 shrinkage + plasticity after advantage computed (below)
            new_ws.append((i, wi, drive))

        # update substrate with pump + a tiny random motor push inside band
        mot = np.zeros_like(s_dec)
        noise_push = self.rng.normal(0.0, c.motor_noise, size=band)
        mot[:band] += noise_push

        cur = s_dec + pump + mot
        self.S[t] = cur

        # compute advantage and do weight updates now that S is updated
        E_cell = float(np.mean(cur))
        advantage = E_cell - E_cell_prev

        for i, wi, drive in new_ws:
            # gradient: centered env (local, no backprop)
            grad_w = advantage * centered
            wi = wi + c.eta * grad_w - c.lam_l1 * np.sign(wi)
            self.w[i] = wi
            self.age[i] += 1
            self.energy_accum[i] += advantage

        # boundary motion from kernels + tension – cost
        delta_b = c.alpha_move * total_motion - c.beta_tension * (self.b - np.round(self.b))
        delta_b += self.rng.normal(0.0, c.motor_noise)
        move_cost = c.c_motor * abs(delta_b)
        self.b = (self.b + delta_b) % self.X_env

        # basic growth/prune
        if (len(self.w) < c.max_conns) and (self.rng.random() < c.spawn_rate):
            self._spawn_kernel()
        # prune stale underperformers
        keep = []
        for i in range(len(self.w)):
            age_ok = self.age[i] >= c.min_age
            weak = (abs(self.energy_accum[i]) < c.prune_thresh) and (np.sum(np.abs(self.w[i])) < 1.0)
            if age_ok and weak:
                continue
            keep.append(i)
        if len(keep) != len(self.w):
            self.w = [self.w[i] for i in keep]
            self.age = [self.age[i] for i in keep]
            self.energy_accum = [self.energy_accum[i] for i in keep]
            self.motion_accum = [self.motion_accum[i] for i in keep]
            self.harv_accum = [self.harv_accum[i] for i in keep]

        # bookkeeping (note: E_flux = total pump from boundary)
        E_env = float(np.mean(e_row))
        E_flux = float(np.sum(pump))

        # subtract motion cost from cell energy record (visual accounting)
        E_cell_adj = E_cell - move_cost

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell_adj)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)

        return E_cell_adj, E_env, E_flux

    # ----- run -----

    def run(self, progress_cb: Optional[Callable[[int], None]] = None,
            snapshot_every: Optional[int] = None):
        snap = int(snapshot_every) if snapshot_every is not None else 0
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                if snap > 0:
                    if (t % snap) == 0 or (t == self.T - 1):
                        progress_cb(t)
                else:
                    progress_cb(t)


# ----------------------------
# Helpers for the app
# ----------------------------

def default_config() -> Dict:
    return asdict(Config())

def make_engine(cfg_dict: Dict) -> Engine:
    # Robustly build dataclasses from a plain dict
    env_in = cfg_dict.get("env", {})
    fcfg = FieldCfg(
        length=int(env_in.get("length", 512)),
        frames=int(cfg_dict.get("frames", 5000)),
        noise_sigma=float(env_in.get("noise_sigma", 0.01)),
        sources=env_in.get("sources", FieldCfg().sources),
    )
    cfg = Config(
        seed=int(cfg_dict.get("seed", 0)),
        frames=int(cfg_dict.get("frames", 5000)),
        space=int(cfg_dict.get("space", 64)),
        band=int(cfg_dict.get("band", 2)),
        diffuse=float(cfg_dict.get("diffuse", 0.15)),
        decay=float(cfg_dict.get("decay", 0.01)),
        k_flux=float(cfg_dict.get("k_flux", 0.08)),
        k_motor=float(cfg_dict.get("k_motor", 1.0)),
        motor_noise=float(cfg_dict.get("motor_noise", 0.02)),
        c_motor=float(cfg_dict.get("c_motor", 0.80)),
        alpha_move=float(cfg_dict.get("alpha_move", 0.3)),
        beta_tension=float(cfg_dict.get("beta_tension", 0.02)),
        gate_win=int(cfg_dict.get("gate_win", 30)),
        eta=float(cfg_dict.get("eta", 0.02)),
        ema_beta=float(cfg_dict.get("ema_beta", 0.10)),
        lam_l1=float(cfg_dict.get("lam_l1", 0.10)),
        prune_thresh=float(cfg_dict.get("prune_thresh", 0.10)),
        min_age=int(cfg_dict.get("min_age", 200)),
        spawn_rate=float(cfg_dict.get("spawn_rate", 0.02)),
        max_conns=int(cfg_dict.get("max_conns", 64)),
        env=fcfg,
    )
    return Engine(cfg)