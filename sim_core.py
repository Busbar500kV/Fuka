# sim_core.py
# A compact, robust baseline that won't NaN/overflow on Streamlit Cloud.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np

# ---------- helpers (numerically safe) ----------
def relu(x):            # piecewise-linear; no exp overflows
    return np.where(x > 0, x, 0.0)

def softclip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def lin_kernel(dist, scale):
    # local, compact support kernel (triangle) — no trig
    # height=1 at 0, goes to 0 at dist>=scale
    s = np.maximum(scale, 1e-8)
    k = np.maximum(0.0, 1.0 - dist / s)
    return k

# ---------- model ----------
@dataclass
class Config:
    frames: int = 1600          # total time
    space: int = 192            # 1D spatial sites
    seed: int = 0

    # environment / free-energy knobs
    env_baseline: float = 0.00  # constant background free energy
    env_pulse_rate: float = 0.002   # chance per frame to spawn an event
    env_pulse_E: float = 3.0    # energy in each event
    env_boundary_boost: float = 0.0 # extra free energy near boundaries (motors!)

    # substrate dynamics
    decay_subs: float = 0.005   # passive loss / dissipation per frame
    diffuse_subs: float = 0.12  # 1D diffusion strength (0..0.5)

    # connections (generic; roles emerge)
    init_n: int = 9             # initial number of connections
    grow_every: int = 200       # try growth this often
    grow_budget: int = 3        # attempts per growth step
    prune_every: int = 200      # try prune this often

    # costs & harvest weights (dimensionless)
    gamma_direct: float = 1.0
    gamma_indirect: float = 0.5
    gamma_motor: float = 1.0

    cost_activation: float = 0.02
    cost_maintenance: float = 0.03
    cost_plasticity: float = 0.005

    # mutation / plasticity noise
    sigma_ell: float = 0.02
    sigma_span: float = 0.06

    # growth seeding
    ell_prior_lo: float = 0.0   # in [0,1] along space
    ell_prior_hi: float = 1.0
    span_prior_lo: float = 0.05 # spatial scale for local kernels
    span_prior_hi: float = 0.25

    # safety clamps
    max_energy_abs: float = 1e6

def default_config():
    return asdict(Config())

# ---------- environment ----------
def step_environment(rng, env_t: np.ndarray, cfg: Config) -> np.ndarray:
    """
    env_t: (X,) previous environment energy field
    Returns new env field with occasional pulses and boundary boosts.
    """
    X = env_t.shape[0]
    env = env_t.copy()

    # background/boundary supply (no trig; just constants + spatial ramp near edges)
    if cfg.env_baseline != 0.0:
        env += cfg.env_baseline

    if cfg.env_boundary_boost != 0.0:
        x = np.linspace(0, 1, X)
        edge = np.minimum(x, 1 - x)  # 0 at boundary, 0.5 in middle
        boost = cfg.env_boundary_boost * (0.5 - edge) * 2.0 # linear ramp → big at edges
        env += boost

    # random local "events" (space-time pulses) that inject free energy
    if rng.random() < cfg.env_pulse_rate:
        center = int(rng.integers(0, X))
        span   = max(2, int(rng.integers(6, 18)))
        lo, hi = max(0, center - span), min(X, center + span + 1)
        ramp = 1.0 - np.abs(np.linspace(lo, hi-1, hi-lo) - center) / (span + 1e-6)
        env[lo:hi] += cfg.env_pulse_E * ramp

    # keep positive (environment is a source)
    return np.maximum(env, 0.0)

# ---------- connections ----------
class Connection:
    """
    Generic connection defined by (ell, span). No hard-coded role.
    - Direct harvest uses env (SENSE-like).
    - Indirect harvest uses substrate (INTERNAL-like).
    - Motor harvest uses 'pushing' substrate toward env energy (MOTOR-like).
    The dominant contributor over its lifetime defines the emergent role.
    """
    __slots__ = ("ell","span","age","energy_contrib",
                 "harv_direct","harv_indirect","harv_motor","active_frac")

    def __init__(self, ell: float, span: float):
        self.ell = float(softclip(ell, 0.0, 1.0))
        self.span = float(softclip(span, 0.01, 1.0))
        self.age = 0
        self.energy_contrib = 0.0
        self.harv_direct = 0.0
        self.harv_indirect = 0.0
        self.harv_motor = 0.0
        self.active_frac = 0.0

    def kernel(self, X: int) -> np.ndarray:
        x = np.linspace(0, 1, X)
        dist = np.abs(x - self.ell)
        return lin_kernel(dist, self.span)

    def propose_mutation(self, rng: np.random.Generator, cfg: Config) -> Tuple[float,float]:
        ell = softclip(self.ell + cfg.sigma_ell * rng.normal(), 0.0, 1.0)
        span = softclip(self.span + cfg.sigma_span * rng.normal(), 0.01, 1.0)
        return ell, span

# ---------- simulation core ----------
def run_sim(cfg_dict: Dict) -> Dict:
    cfg = Config(**cfg_dict)
    rng = np.random.default_rng(cfg.seed)

    X, T = cfg.space, cfg.frames
    # state arrays
    env_hist = np.zeros((X, T), dtype=np.float32)
    subs_hist = np.zeros((X, T), dtype=np.float32)
    conn_hist = []  # will record activation per-conn per-time (list of arrays)

    # global energy (may go negative; clamp display later)
    E = 50.0

    # substrate S (what the "cell" internal field stores)
    S = np.zeros(X, dtype=np.float32)
    env = np.zeros(X, dtype=np.float32)

    # initialize connections
    conns: List[Connection] = []
    for _ in range(cfg.init_n):
        ell = float(rng.uniform(cfg.ell_prior_lo, cfg.ell_prior_hi))
        span = float(rng.uniform(cfg.span_prior_lo, cfg.span_prior_hi))
        conns.append(Connection(ell, span))

    # record activations over time for plotting
    act_rows = np.zeros((cfg.init_n, T), dtype=np.float32)

    # run loop
    for t in range(T):
        # 1) environment update (source)
        env = step_environment(rng, env, cfg)

        # 2) each connection acts locally with kernel k
        #    - direct harvest from env via match to local gradient in env
        #    - indirect harvest from stored substrate when pattern matches
        #    - motor "pushing": moving substrate toward env (pays act cost; returns if moves energy inward)
        if t == 0:
            conn_hist = []

        frame_acts = []
        E_delta = 0.0

        # cheap 1D diffusion for substrate before acting (passive physics)
        if cfg.diffuse_subs > 0:
            S = diffuse1d(S, cfg.diffuse_subs)

        # passive decay
        if cfg.decay_subs > 0:
            S *= (1.0 - cfg.decay_subs)

        # loop connections
        for i, c in enumerate(conns):
            k = c.kernel(X)
            k = k / (np.sum(k) + 1e-8)  # normalize to keep scales sane

            # signal proxies (no trig):
            # local mismatch (how far env is from subs)
            mismatch = np.abs(env - S)

            # direct harvest: high when env has "sharp" local structure under k
            local_env = np.sum(k * env)
            local_var_env = np.sum(k * np.abs(env - local_env))
            P_dir = cfg.gamma_direct * local_var_env

            # indirect harvest: high when subs has compact structure (memory reuse)
            local_sub = np.sum(k * S)
            local_var_sub = np.sum(k * np.abs(S - local_sub))
            P_ind = cfg.gamma_indirect * local_var_sub

            # motor work: move a bit of subs toward env under kernel (local action)
            # Try a tiny step; result energy gain equals reduction in mismatch (proxy)
            step = 0.15 * (env - S)
            delta = np.sum(k * (np.abs(mismatch) - np.abs(env - (S + step))))
            P_mot = cfg.gamma_motor * np.maximum(0.0, delta)
            # Apply the motor step to substrate (local)
            S = S + k * step

            # costs (local, simple)
            C_act = cfg.cost_activation * (np.sum(k) + local_var_env + local_var_sub)
            C_main = cfg.cost_maintenance * (1.0 + 1.0 / (c.span + 1e-6))

            net = (P_dir + P_ind + P_mot) - (C_act + C_main)
            E_delta += float(net)
            c.energy_contrib += float(net)
            c.harv_direct += float(P_dir)
            c.harv_indirect += float(P_ind)
            c.harv_motor += float(P_mot)
            c.age += 1
            c.active_frac = (c.active_frac * (c.age - 1) + (net > 0)) / c.age

            frame_acts.append(net)
            # store activation time series (grow matrix if we add connections later)
            if i >= act_rows.shape[0]:
                act_rows = pad_rows(act_rows, i + 1)
            act_rows[i, t] = net

        # 3) update global energy
        E += E_delta
        E = float(np.clip(E, -cfg.max_energy_abs, cfg.max_energy_abs))

        # 4) growth & prune (local, greedy on energy improvement)
        if (t > 0) and (t % cfg.grow_every == 0):
            for _ in range(cfg.grow_budget):
                # seed at random spatial location
                ell0 = float(rng.uniform(cfg.ell_prior_lo, cfg.ell_prior_hi))
                span0 = float(rng.uniform(cfg.span_prior_lo, cfg.span_prior_hi))
                conns.append(Connection(ell0, span0))
                # ensure matrix space
                if len(conns) > act_rows.shape[0]:
                    act_rows = pad_rows(act_rows, len(conns))

        if (t > 0) and (t % cfg.prune_every == 0) and len(conns) > 4:
            # prune the weakest contributors
            energies = np.array([c.energy_contrib for c in conns])
            worst_idx = np.argsort(energies)[:max(1, len(conns)//10)]
            mask = np.ones(len(conns), dtype=bool)
            mask[worst_idx] = False
            conns = [c for c, keep in zip(conns, mask) if keep]

        # 5) write histories
        env_hist[:, t] = env
        subs_hist[:, t] = S

    # ----- role summary (emergent) -----
    roles = summarize_roles(conns)

    return dict(
        env=env_hist,
        subs=subs_hist,
        acts=act_rows[:len(conns), :],
        energy=float(E),
        roles=roles,
        table=make_table(conns)
    )

def diffuse1d(S: np.ndarray, alpha: float) -> np.ndarray:
    """ Simple reflecting-boundary diffusion """
    X = len(S)
    out = S.copy()
    a = softclip(alpha, 0.0, 0.49)
    # neighbors
    left = np.roll(S, 1); left[0] = S[0]
    right = np.roll(S, -1); right[-1] = S[-1]
    out = (1 - 2*a) * S + a * (left + right)
    return out

def pad_rows(M: np.ndarray, rows: int) -> np.ndarray:
    if M.shape[0] >= rows:
        return M
    extra = np.zeros((rows - M.shape[0], M.shape[1]), dtype=M.dtype)
    return np.vstack([M, extra])

def summarize_roles(conns: List[Connection]) -> Dict[str, int]:
    counts = dict(SENSE=0, INTERNAL=0, MOTOR=0, UNLABELED=0)
    for c in conns:
        trio = np.array([c.harv_direct, c.harv_indirect, c.harv_motor])
        if not np.isfinite(trio).all() or trio.sum() <= 0:
            counts["UNLABELED"] += 1
            continue
        label = ["SENSE", "INTERNAL", "MOTOR"][int(np.argmax(trio))]
        counts[label] += 1
    return counts

def make_table(conns: List[Connection]) -> List[Dict]:
    rows = []
    for i, c in enumerate(conns):
        rows.append(dict(
            index=i,
            type="",  # emergent; left blank (use roles chart)
            L=float(c.ell),
            span=float(c.span),
            Active_Frac=float(c.active_frac),
            Contribution_Energy=float(c.energy_contrib),
            Harv_Direct=float(c.harv_direct),
            Harv_Indirect=float(c.harv_indirect),
            Harv_Motor=float(c.harv_motor),
            Age=int(c.age),
        ))
    return rows