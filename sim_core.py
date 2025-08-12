# sim_core.py
# Stream-friendly engine with local physics + a learnable gate kernel.

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


# =========================
# Config data classes
# =========================

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 5000
    noise_sigma: float = 0.00
    baseline: float = 0.0
    amp_scale: float = 1.0
    # Each: {"kind":"moving_peak","amp":float,"speed":float,"width":float,"start":int}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 5.0, "start": 15}
    ])


@dataclass
class KernelCfg:
    enabled: bool = True
    radius: int = 3
    lr: float = 1e-3
    l2: float = 1e-4
    init: float = 0.0
    k_gate: float = 0.15


@dataclass
class Thresholds:
    grow: float = 0.6
    prune: float = 0.1


@dataclass
class GrowthCfg:
    rate: float = 0.0
    prune_rate: float = 0.0


@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64

    k_flux: float = 0.08
    k_motor: float = 0.50
    band: int = 3
    decay: float = 0.01
    diffuse: float = 0.12
    k_noise: float = 0.00
    cap: float = 5.0

    boundary_speed: float = 0.04

    kernel: KernelCfg = field(default_factory=KernelCfg)
    growth: GrowthCfg = field(default_factory=GrowthCfg)
    thresholds: Thresholds = field(default_factory=Thresholds)

    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Environment builder
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))
    return amp * np.exp(-(d**2) / (2.0 * max(1e-6, width)**2))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)

    for s in cfg.sources:
        kind = s.get("kind", "moving_peak")
        if kind != "moving_peak":
            continue
        amp   = float(s.get("amp", 1.0)) * float(cfg.amp_scale)
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X
        speed_frac = float(s.get("speed", 0.0))
        speed = speed_frac * X

        pos = float(start)
        for t in range(T):
            E[t] += _moving_peak(X, pos, amp, width)
            pos = (pos + speed) % X

    if cfg.baseline != 0.0:
        E += float(cfg.baseline)
    if cfg.noise_sigma > 0.0:
        E += rng.normal(0.0, cfg.noise_sigma, size=E.shape)

    np.maximum(E, 0.0, out=E)
    return E


# =========================
# History
# =========================

@dataclass
class History:
    t: List[int] = field(default_factory=list)
    E_cell: List[float] = field(default_factory=list)
    E_env:  List[float] = field(default_factory=list)
    E_flux: List[float] = field(default_factory=list)


# =========================
# Engine
# =========================

class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.env = build_env(cfg.env, self.rng)   # (T_env, X_env)
        self.T = int(cfg.frames)
        self.X = int(cfg.space)

        self.S = np.zeros((self.T, self.X), dtype=float)

        r = max(0, int(cfg.kernel.radius))
        ksize = 2 * r + 1
        if cfg.kernel.init > 0.0:
            self.w = self.rng.normal(0.0, cfg.kernel.init, size=ksize).astype(float)
        else:
            self.w = np.zeros(ksize, dtype=float)
        self.w[r] = 1.0

        self.offset = 0.0
        self.hist = History()

    def _sample_env_row(self, t: int) -> np.ndarray:
        E = self.env
        X_env = E.shape[1]
        base_idx = (np.arange(self.X) * X_env // self.X)
        idx = (base_idx + int(self.offset)) % X_env
        return E[min(t, E.shape[0]-1), idx]

    def _convolve_same(self, signal: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = (len(w) - 1) // 2
        pad = np.pad(signal, (r, r), mode='reflect')
        return np.convolve(pad, w, mode='valid')

    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        e_row = self._sample_env_row(t)

        prev = self.S[t-1] if t > 0 else self.S[0]

        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5*(left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        band = max(1, int(cfg.band))
        grad = np.maximum(e_row[:band] - S_decayed[:band], 0.0)
        pump = np.zeros_like(S_decayed)
        pump[:band] += cfg.k_flux * grad

        mot = np.zeros_like(S_decayed)
        if cfg.k_motor > 0.0:
            mot[:band] += cfg.k_motor * self.rng.random(band)

        gated = np.zeros_like(S_decayed)
        if cfg.kernel.enabled and len(self.w) > 0 and cfg.kernel.k_gate != 0.0:
            gated = cfg.kernel.k_gate * self._convolve_same(S_decayed, self.w)

        noise = 0.0
        if cfg.k_noise > 0.0:
            noise = cfg.k_noise * self.rng.normal(0.0, 1.0, size=S_decayed.shape)

        cur = S_decayed + pump + mot + gated + noise
        if cfg.cap is not None and cfg.cap > 0:
            np.clip(cur, 0.0, float(cfg.cap), out=cur)
        self.S[t] = cur

        # kernel plasticity (local correlation)
        if cfg.kernel.enabled and len(self.w) > 0 and cfg.kernel.lr > 0.0:
            r = (len(self.w) - 1) // 2
            win = min(4 * (r + 1), self.X)
            sig = S_decayed[:win]
            err = (e_row[:win] - cur[:win])
            if len(sig) >= (2*r+1):
                patches = []
                for i in range(r, len(sig)-r):
                    patches.append(sig[i-r:i+r+1])
                if patches:
                    P = np.asarray(patches)  # [n, 2r+1]
                    E = err[r:len(sig)-r]    # [n]
                    grad_w = (P * E[:, None]).mean(axis=0) - cfg.kernel.l2 * self.w
                    self.w += cfg.kernel.lr * grad_w
                    nrm = np.linalg.norm(self.w)
                    if nrm > 0:
                        self.w /= (1e-6 + nrm)

        motor_mass = float(np.sum(mot[:band]))
        self.offset = (self.offset + cfg.boundary_speed * motor_mass) % self.env.shape[1]

        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump))

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)

        return E_cell, E_env, E_flux

    def run(self, progress_cb: Optional[Callable[[int], None]] = None, snapshot_every: Optional[int] = None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)


# =========================
# Public helpers
# =========================

def default_config() -> Dict:
    return asdict(Config())


def make_config_from_dict(cfg: Dict) -> Config:
    """Build a Config from a plain dict (Streamlit state)."""
    env_in = cfg.get("env", {})
    sources = env_in.get("sources", FieldCfg().sources)

    fcfg = FieldCfg(
        length=int(env_in.get("length", 512)),
        frames=int(env_in.get("frames", int(cfg.get("frames", 5000)))),
        noise_sigma=float(env_in.get("noise_sigma", 0.0)),
        baseline=float(env_in.get("baseline", 0.0)),
        amp_scale=float(env_in.get("amp_scale", 1.0)),
        sources=sources,
    )
    k_in = cfg.get("kernel", {})
    kcfg = KernelCfg(
        enabled=bool(k_in.get("enabled", True)),
        radius=int(k_in.get("radius", 3)),
        lr=float(k_in.get("lr", 1e-3)),
        l2=float(k_in.get("l2", 1e-4)),
        init=float(k_in.get("init", 0.0)),
        k_gate=float(k_in.get("k_gate", 0.15)),
    )
    th_in = cfg.get("thresholds", {})
    gc_in = cfg.get("growth", {})

    ecfg = Config(
        seed=int(cfg.get("seed", 0)),
        frames=int(cfg.get("frames", 5000)),
        space=int(cfg.get("space", 64)),
        k_flux=float(cfg.get("k_flux", 0.08)),
        k_motor=float(cfg.get("k_motor", 0.50)),
        band=int(cfg.get("band", 3)),
        decay=float(cfg.get("decay", 0.01)),
        diffuse=float(cfg.get("diffuse", 0.12)),
        k_noise=float(cfg.get("k_noise", 0.00)),
        cap=float(cfg.get("cap", 5.0)),
        boundary_speed=float(cfg.get("boundary_speed", 0.04)),
        kernel=kcfg,
        growth=GrowthCfg(
            rate=float(gc_in.get("rate", 0.0)),
            prune_rate=float(gc_in.get("prune_rate", 0.0)),
        ),
        thresholds=Thresholds(
            grow=float(th_in.get("grow", 0.6)),
            prune=float(th_in.get("prune", 0.1)),
        ),
        env=fcfg,
    )
    return ecfg