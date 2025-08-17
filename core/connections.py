# core/connections.py
# Minimal, self-contained "connections" layer (no UI, no engine loop)
# - Defines the ConnectionState container
# - Growth / pruning / plasticity
# - Rendering of local sampling kernels for the physics layer

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


# ----------------------------
# Data containers
# ----------------------------

@dataclass
class ConnectionState:
    """
    Vectorized store of 'world-line' connections that live on a 1D ring (space length = X).
    All arrays are length N (number of connections).
      x        : integer center position (0..X-1)
      ell      : spatial scale / kernel width (>= 0)
      tau      : temporal inertia / time-scale knob (>= 0). Physics chooses how to use it.
      energy   : current internal energy carried by the connection
      age      : integer age in frames
      alive    : boolean mask; connections with alive==False are empty slots (can be reused)
      tag      : optional integer tag for emergent roles (0=generic). NOT used by physics here.
    """
    X: int
    N: int
    x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    ell: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    age: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    alive: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    tag: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    def __post_init__(self):
        # Allocate arrays if not provided
        if self.x.size == 0:
            self.x = np.zeros(self.N, dtype=np.int32)
        if self.ell.size == 0:
            self.ell = np.zeros(self.N, dtype=np.float64)
        if self.tau.size == 0:
            self.tau = np.zeros(self.N, dtype=np.float64)
        if self.energy.size == 0:
            self.energy = np.zeros(self.N, dtype=np.float64)
        if self.age.size == 0:
            self.age = np.zeros(self.N, dtype=np.int32)
        if self.alive.size == 0:
            self.alive = np.zeros(self.N, dtype=bool)
        if self.tag.size == 0:
            self.tag = np.zeros(self.N, dtype=np.int8)

        # Clamp and wrap
        self.x %= max(1, self.X)
        np.maximum(self.ell, 0.0, out=self.ell)
        np.maximum(self.tau, 0.0, out=self.tau)

    # -------- convenience views --------
    @property
    def count_alive(self) -> int:
        return int(self.alive.sum())

    def free_slots(self) -> np.ndarray:
        """Indices that can be reused to grow new connections."""
        return np.nonzero(~self.alive)[0]

    def alive_idx(self) -> np.ndarray:
        return np.nonzero(self.alive)[0]


# ----------------------------
# Construction / initialization
# ----------------------------

def init_connections(X: int,
                     N: int,
                     rng: np.random.Generator,
                     n_seed: int = 0,
                     seed_positions: Optional[np.ndarray] = None,
                     ell_prior: Tuple[float, float] = (0.5, 2.0),
                     tau_prior: Tuple[float, float] = (0.1, 1.0),
                     energy0: float = 0.0) -> ConnectionState:
    """
    Create a ConnectionState with capacity N. Optionally seed with n_seed alive connections.
    If seed_positions is provided (array of ints), they’re used (wrapped to X).
    """
    st = ConnectionState(X=X, N=N)
    if n_seed <= 0:
        return st

    slots = np.arange(min(n_seed, N))
    st.alive[slots] = True
    if seed_positions is not None and seed_positions.size >= len(slots):
        st.x[slots] = np.asarray(seed_positions[:len(slots)], dtype=np.int64) % X
    else:
        st.x[slots] = rng.integers(0, X, size=len(slots))

    st.ell[slots] = rng.uniform(ell_prior[0], ell_prior[1], size=len(slots))
    st.tau[slots] = rng.uniform(tau_prior[0], tau_prior[1], size=len(slots))
    st.energy[slots] = energy0
    st.age[slots] = 0
    st.tag[slots] = 0
    return st


# ----------------------------
# Growth / pruning / plasticity
# ----------------------------

def grow_connections(state: ConnectionState,
                     rng: np.random.Generator,
                     budget: int,
                     seed_bias: Optional[np.ndarray] = None,
                     ell_prior: Tuple[float, float] = (0.5, 2.0),
                     tau_prior: Tuple[float, float] = (0.1, 1.0),
                     energy0: float = 0.0) -> int:
    """
    Try to grow up to 'budget' new alive connections into free slots.
      seed_bias (optional): length X nonnegative weights to bias positions (e.g., around a boundary).
    Returns number of connections actually created.
    """
    free = state.free_slots()
    if free.size == 0 or budget <= 0:
        return 0
    k = int(min(budget, free.size))
    slots = free[:k]

    if seed_bias is None:
        xs = rng.integers(0, state.X, size=k)
    else:
        w = np.array(seed_bias, dtype=np.float64).clip(min=0.0)
        if not np.any(w > 0):
            w[:] = 1.0  # uniform fallback
        w = w / w.sum()
        xs = rng.choice(state.X, size=k, replace=True, p=w)

    state.alive[slots] = True
    state.x[slots] = xs
    state.ell[slots] = rng.uniform(ell_prior[0], ell_prior[1], size=k)
    state.tau[slots] = rng.uniform(tau_prior[0], tau_prior[1], size=k)
    state.energy[slots] = energy0
    state.age[slots] = 0
    state.tag[slots] = 0
    return k


def prune_connections(state: ConnectionState,
                      min_age: int,
                      energy_thresh: float) -> int:
    """
    Kill connections whose age >= min_age and energy < energy_thresh.
    Returns number pruned.
    """
    idx = state.alive_idx()
    if idx.size == 0:
        return 0
    kill = idx[(state.age[idx] >= min_age) & (state.energy[idx] < energy_thresh)]
    if kill.size:
        state.alive[kill] = False
        # zero out for cleanliness (not required)
        state.energy[kill] = 0.0
        state.ell[kill] = 0.0
        state.tau[kill] = 0.0
        state.tag[kill] = 0
        state.age[kill] = 0
    return int(kill.size)


def plasticity_step(state: "ConnectionState",
                    rng: np.random.Generator,
                    sigma_ell: float = 0.05,
                    sigma_tau: float = 0.05,
                    p_move: float = 0.05,
                    env_row: np.ndarray | None = None,
                    subs_row: np.ndarray | None = None,
                    **kwargs) -> None:
    """
    Small local mutations on alive connections:
      - ell, tau jitter (non-negative clamp)
      - with prob p_move, shift x by ±1 (ring)

    If env_row is provided, we (optionally) bias the one-step move toward
    higher env locally. If subs_row is provided, we can damp moves in
    already-high substrate (prevent thrashing). If neither is provided,
    behavior is identical to the old version.
    """
    idx = state.alive_idx()
    if idx.size == 0:
        return

    # width/inertia jitter
    state.ell[idx] = np.maximum(0.0, state.ell[idx] + sigma_ell * rng.normal(size=idx.size))
    state.tau[idx] = np.maximum(0.0, state.tau[idx] + sigma_tau * rng.normal(size=idx.size))

    # occasional one-step move along the ring
    move_mask = rng.random(idx.size) < p_move
    if not np.any(move_mask):
        return

    # default unbiased steps: -1, 0, +1
    steps = rng.integers(-1, 2, size=move_mask.sum())

    if env_row is not None and env_row.size == state.X:
        pos = state.x[idx[move_mask]]
        left_pos  = (pos - 1) % state.X
        right_pos = (pos + 1) % state.X

        cur_e   = env_row[pos]
        left_e  = env_row[left_pos]
        right_e = env_row[right_pos]

        # local preference for higher env vs current
        p_left  = (left_e  > cur_e).astype(float)
        p_right = (right_e > cur_e).astype(float)
        p_stay  = 1.0 - np.maximum(p_left, p_right) * 0.5  # keep some stay prob

        # optionally damp movement where substrate is already high
        if subs_row is not None and subs_row.size == state.X:
            cur_s = subs_row[pos]
            # Larger substrate → more "stickiness"
            stick = 1.0 / (1.0 + cur_s)
            p_left  *= stick
            p_right *= stick
            # keep p_stay as-is so stickiness increases staying

        psum = p_left + p_right + p_stay
        # normalize or fallback to uniform if degenerate
        ok = psum > 0
        p_left  = np.where(ok, p_left/psum, 1/3)
        p_stay  = np.where(ok, p_stay/psum, 1/3)
        p_right = np.where(ok, p_right/psum, 1/3)

        r = rng.random(move_mask.sum())
        steps = np.where(r < p_left, -1,
                 np.where(r < p_left + p_stay, 0, +1))

    state.x[idx[move_mask]] = (state.x[idx[move_mask]] + steps) % state.X

def age_and_dissipate(state: ConnectionState,
                      dissipation: float = 0.0) -> None:
    """
    Increment age and optionally bleed some internal energy (simple decay).
    """
    idx = state.alive_idx()
    if idx.size == 0:
        return
    state.age[idx] += 1
    if dissipation > 0.0:
        state.energy[idx] = np.maximum(0.0, state.energy[idx] * (1.0 - dissipation))


# ----------------------------
# Kernel rendering (for physics)
# ----------------------------

def gaussian_kernels(state: ConnectionState,
                     X: Optional[int] = None,
                     clip_radius: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sampling kernels for *alive* connections as a dense matrix W [M, X],
    where M = number of alive connections (order follows alive_idx()).

    Each row m is a circular 1D Gaussian centered at x_m with width ell_m.
    Kernels are L1-normalized per row (sum to 1) so physics can use them as weights.

    Returns:
      (idx, W)
        idx : int array of alive connection indices (into state arrays)
        W   : float array [M, X] with nonnegative weights; rows sum to 1 (unless ell==0 -> delta)
    """
    if X is None:
        X = state.X
    idx = state.alive_idx()
    M = idx.size
    if M == 0:
        return idx, np.zeros((0, X), dtype=np.float64)

    centers = state.x[idx].astype(np.float64)
    widths  = np.maximum(1e-6, state.ell[idx].astype(np.float64))  # avoid div by zero
    xs = np.arange(X, dtype=np.float64)[None, :]                    # [1, X]
    cs = centers[:, None]                                           # [M, 1]

    # circular distance on the ring
    d = np.abs(xs - cs)                                             # [M, X]
    d = np.minimum(d, X - d)

    if clip_radius is not None and clip_radius > 0:
        # mask far points to zero to keep kernels compact (optional optimization)
        mask = d > clip_radius
    else:
        mask = None

    # Gaussian weights
    # w_ij ∝ exp(-(d^2)/(2 σ^2)), with σ = widths
    sig2 = 2.0 * (widths[:, None] ** 2)
    W = np.exp(- (d ** 2) / sig2)

    if mask is not None:
        W[mask] = 0.0

    # Row-normalize (L1) — if a row is all-zero, fallback to a delta at the center
    row_sum = W.sum(axis=1, keepdims=True)
    zero_rows = (row_sum[:, 0] <= 1e-12)
    if np.any(zero_rows):
        W[zero_rows, :] = 0.0
        W[zero_rows, state.x[idx[zero_rows]]] = 1.0
        row_sum = W.sum(axis=1, keepdims=True)

    W = W / row_sum
    return idx, W


# ----------------------------
# Diagnostics / export helpers
# ----------------------------

def as_table(state: ConnectionState) -> Dict[str, np.ndarray]:
    """Return a dict of arrays for logging/inspection."""
    return {
        "x": state.x.copy(),
        "ell": state.ell.copy(),
        "tau": state.tau.copy(),
        "energy": state.energy.copy(),
        "age": state.age.copy(),
        "alive": state.alive.copy(),
        "tag": state.tag.copy(),
    }


def summary(state: ConnectionState) -> Dict[str, float]:
    """Small numeric summary useful for UI."""
    idx = state.alive_idx()
    return {
        "alive": float(idx.size),
        "mean_ell": float(np.mean(state.ell[idx])) if idx.size else 0.0,
        "mean_tau": float(np.mean(state.tau[idx])) if idx.size else 0.0,
        "mean_energy": float(np.mean(state.energy[idx])) if idx.size else 0.0,
        "max_energy": float(np.max(state.energy[idx])) if idx.size else 0.0,
        "min_energy": float(np.min(state.energy[idx])) if idx.size else 0.0,
    }