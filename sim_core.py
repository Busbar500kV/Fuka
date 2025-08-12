# sim_core.py
# Minimal, stream-friendly engine with local physics + a learnable gate kernel.

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


# =========================
# Config data classes
# =========================

@dataclass
class FieldCfg:
    """Environment field (space-time) configuration."""
    length: int = 512            # env spatial length
    frames: int = 5000           # env time length (usually match run frames)
    noise_sigma: float = 0.00    # small env noise
    baseline: float = 0.0        # constant offset added everywhere
    amp_scale: float = 1.0       # multiply all sources by this
    # Each source: {"kind":"moving_peak","amp":float,"speed":float,"width":float,"start":int}
    # speed is in fractions of env length per frame (we convert to cells/frame)
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 5.0, "start": 15}
    ])


@dataclass
class KernelCfg:
    enabled: bool = True
    radius: int = 3              # kernel half-width -> kernel size = 2r+1
    lr: float = 1e-3             # plasticity rate
    l2: float = 1e-4             # weight decay
    init: float = 0.0            # initial weights ~ N(0, init)
    k_gate: float = 0.15         # how much gated signal adds to substrate


@dataclass
class Thresholds:
    grow: float = 0.6
    prune: float = 0.1


@dataclass
class GrowthCfg:
    rate: float = 0.0   # placeholder for future structural growth
    prune_rate: float = 0.0


@dataclass
class Config:
    # run
    seed: int = 0
    frames: int = 5000
    space: int = 64

    # energy / dynamics knobs
    k_flux: float = 0.08        # env->cell pump strength in boundary band
    k_motor: float = 0.50       # random motor drive in band
    band: int = 3               # width of boundary band (cells)
    decay: float = 0.01         # substrate local decay per step
    diffuse: float = 0.12       # substrate diffusion strength
    k_noise: float = 0.00       # substrate local noise
    cap: float = 5.0            # per-cell saturation cap (clip after update)

    # boundary movement
    boundary_speed: float = 0.04  # how fast offset moves per unit motor mass

    # kernel / “general connections”
    kernel: KernelCfg = field(default_factory=KernelCfg)

    # growth / prune (placeholders, not used heavily yet)
    growth: GrowthCfg = field(default_factory=GrowthCfg)
    thresholds: Thresholds = field(default_factory=Thresholds)

    # environment
    env: FieldCfg = field(default_factory=FieldCfg)


# =========================
# Environment builder
# =========================

def _moving_peak(space: int, pos: float, amp: float, width: float) -> np.ndarray:
    x = np.arange(space, dtype=float)
    d = np.minimum(np.abs(x - pos), space - np.abs(x - pos))  # wrap-around distance
    return amp * np.exp(-(d ** 2) / (2.0 * max(1e-6, width) ** 2))


def build_env(cfg: FieldCfg, rng: np.random.Generator) -> np.ndarray:
    """Create env field E[t, x] with all sources."""
    T, X = cfg.frames, cfg.length
    E = np.zeros((T, X), dtype=float)

    for s in cfg.sources:
        kind = s.get("kind", "moving_peak")
        amp = float(s.get("amp", 1.0)) * float(cfg.amp_scale)
        width = float(s.get("width", 4.0))
        start = int(s.get("start", 0)) % X
        speed_frac = float(s.get("speed", 0.0))               # fraction of X per frame
        speed = speed_frac * X                                 # cells per frame

        if kind != "moving_peak":
            continue

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
# History buffers
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
    """Time-step simulation with local physics + learnable gate kernel."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # env and substrate
        self.env = build_env(cfg.env, self.rng)   # (T_env, X_env)
        self.T = int(cfg.frames)
        self.X = int(cfg.space)

        # substrate S[t, x] (store all frames for plotting)
        self.S = np.zeros((self.T, self.X), dtype=float)

        # learnable gate kernel (odd length)
        r = max(0, int(cfg.kernel.radius))
        ksize = 2 * r + 1
        if cfg.kernel.init > 0.0:
            self.w = self.rng.normal(0.0, cfg.kernel.init, size=ksize).astype(float)
        else:
            self.w = np.zeros(ksize, dtype=float)
        self.w[r] = 1.0  # start as pass-through

        # boundary offset (how far the boundary has “moved” over env space)
        self.offset = 0.0

        # energies (scalars per-step)
        self.hist = History()

    # ---------- helpers ----------

    def _sample_env_row(self, t: int) -> np.ndarray:
        """Sample env[t] onto substrate space with current boundary offset (wrap)."""
        E = self.env
        X_env = E.shape[1]
        # map substrate x -> env index with offset and simple resample
        base_idx = (np.arange(self.X) * X_env // self.X)  # 0..X_env-1 (nearest)
        idx = (base_idx + int(self.offset)) % X_env
        return E[min(t, E.shape[0]-1), idx]

    def _convolve_same(self, signal: np.ndarray, w: np.ndarray) -> np.ndarray:
        # simple 1D same-size convolution (reflect edges)
        r = (len(w) - 1) // 2
        pad = np.pad(signal, (r, r), mode='reflect')
        out = np.convolve(pad, w, mode='valid')
        return out

    # ---------- simulation step ----------

    def step(self, t: int) -> Tuple[float, float, float]:
        cfg = self.cfg

        # env at this step (resampled with boundary offset)
        e_row = self._sample_env_row(t)

        # previous substrate state
        prev = self.S[t-1] if t > 0 else self.S[0]

        # local diffusion + decay
        left  = np.roll(prev,  1)
        right = np.roll(prev, -1)
        S_diffused = prev + cfg.diffuse * (0.5 * (left + right) - prev)
        S_decayed  = (1.0 - cfg.decay) * S_diffused

        # boundary band: env->cell pumping + motor drive
        band = max(1, int(cfg.band))
        grad = np.maximum(e_row[:band] - S_decayed[:band], 0.0)
        pump = np.zeros_like(S_decayed)
        pump[:band] += cfg.k_flux * grad

        mot = np.zeros_like(S_decayed)
        if cfg.k_motor > 0:
            mot[:band] += cfg.k_motor * self.rng.random(band)

        # learned gating (local “general connections”)
        gated = np.zeros_like(S_decayed)
        if cfg.kernel.enabled and len(self.w) > 0 and cfg.kernel.k_gate != 0.0:
            gated = cfg.kernel.k_gate * self._convolve_same(S_decayed, self.w)

        # noise
        noise = 0.0
        if cfg.k_noise > 0.0:
            noise = cfg.k_noise * self.rng.normal(0.0, 1.0, size=S_decayed.shape)

        # update substrate
        cur = S_decayed + pump + mot + gated + noise
        if cfg.cap is not None and cfg.cap > 0:
            np.clip(cur, 0.0, float(cfg.cap), out=cur)
        self.S[t] = cur

        # simple local kernel plasticity: move w to reduce local mismatch
        # mismatch between env and current substrate near boundary
        if cfg.kernel.enabled and len(self.w) > 0 and cfg.kernel.lr > 0.0:
            r = (len(self.w) - 1) // 2
            # use a short window near the boundary where interactions occur
            win = min(4 * (r + 1), self.X)  # small window
            x0, x1 = 0, win
            sig = S_decayed[x0:x1]
            err = (e_row[x0:x1] - cur[x0:x1])  # desire to align with env locally
            # correlate error with local signal to update kernel
            # build a toeplitz-like stack of local patches
            if len(sig) >= (2*r+1):
                patches = []
                for i in range(r, len(sig)-r):
                    patches.append(sig[i-r:i+r+1])
                P = np.asarray(patches)  # [n, 2r+1]
                E = err[r:len(sig)-r]    # [n]
                grad_w = (P * E[:, None]).mean(axis=0) - cfg.kernel.l2 * self.w
                self.w += cfg.kernel.lr * grad_w
                # normalize gently to avoid blow-up
                norm = np.linalg.norm(self.w)
                if norm > 0:
                    self.w /= (1e-6 + norm)

        # boundary movement from motor mass
        motor_mass = float(np.sum(mot[:band]))
        self.offset = (self.offset + cfg.boundary_speed * motor_mass) % self.env.shape[1]

        # bookkeeping energies
        E_cell = float(np.mean(cur))
        E_env  = float(np.mean(e_row))
        E_flux = float(np.sum(pump))

        self.hist.t.append(t)
        self.hist.E_cell.append(E_cell)
        self.hist.E_env.append(E_env)
        self.hist.E_flux.append(E_flux)

        return E_cell, E_env, E_flux

    def run(self, progress_cb: Optional[Callable[[int], None]] = None, snapshot_every: Optional[int] = None):
        """Run full simulation. snapshot_every is accepted for API compatibility (unused here)."""
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)


# =========================
# Public helpers
# =========================

def default_config() -> Dict:
    """Return a plain dict config for Streamlit session_state (JSON-friendly)."""
    return asdict(Config())


def make_config_from_dict(cfg: Dict) -> Config:
    """Safely build Config from a (possibly partial) dict (e.g., Streamlit state)."""
    # env sources may come as text; ensure proper list of dicts
    env_in = cfg.get("env", {})
    sources = env_in.get("sources", FieldCfg().sources)
    # Build nested cfgs
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