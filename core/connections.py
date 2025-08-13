# core/connections.py
import numpy as np
from dataclasses import dataclass, field
from .config import KernelCfg

@dataclass
class GateKernel:
    cfg: KernelCfg
    w: np.ndarray = field(init=False)

    def __post_init__(self):
        r = max(0, int(self.cfg.radius))
        n = 2 * r + 1
        self.w = np.zeros(n, dtype=float)
        self.w[r] = 1.0  # start as local pass-through

    def convolve(self, s: np.ndarray) -> np.ndarray:
        if len(self.w) <= 1:
            return s.copy()
        r = (len(self.w) - 1) // 2
        pad = np.pad(s, (r, r), mode="reflect")
        return np.convolve(pad, self.w, mode="valid")

    def plasticity(self, s_prev: np.ndarray, e_row: np.ndarray, s_cur: np.ndarray):
        # Simple local learning: correlate early-band error with local patches
        if not self.cfg.enabled or self.cfg.lr <= 0.0:
            return
        r = (len(self.w) - 1) // 2
        win = min(4 * (r + 1), len(s_prev))
        if win < (2 * r + 1):
            return
        sig = s_prev[:win]
        err = (e_row[:win] - s_cur[:win])
        patches = [sig[i - r:i + r + 1] for i in range(r, len(sig) - r)]
        if not patches:
            return
        P = np.asarray(patches)               # [n, 2r+1]
        E = err[r:len(sig) - r]               # [n]
        grad = (P * E[:, None]).mean(0) - self.cfg.l2 * self.w
        self.w += self.cfg.lr * grad
        n = np.linalg.norm(self.w)
        if n > 0:
            self.w /= (1e-6 + n)  # keep bounded