# core/connections.py
# Placeholder for future generalized “connections” with growth/prune.
# Keeps the interface ready while the physics remains simple & local.

from dataclasses import dataclass, field
from typing import List

@dataclass
class Connection:
    # generic degrees of freedom; specialize later (sensor/motor/internal emerge)
    strength: float = 0.0
    age: int = 0

@dataclass
class Connectome:
    conns: List[Connection] = field(default_factory=list)

    def grow_random(self, rng, n=1):
        for _ in range(n):
            self.conns.append(Connection(strength=float(rng.random())))
    def prune_weak(self, thr=1e-6):
        self.conns = [c for c in self.conns if c.strength > thr]