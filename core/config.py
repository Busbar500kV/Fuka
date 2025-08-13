# core/config.py
from dataclasses import dataclass, asdict, field
from typing import Dict, List

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 5000
    noise_sigma: float = 0.0
    baseline: float = 0.0
    amp_scale: float = 1.0
    # list of dicts: {"kind":"moving_peak","amp":..., "speed":..., "width":..., "start":...}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 2.0, "speed": 0.00, "width": 6.0, "start": 15}
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
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64
    band: int = 3
    k_flux: float = 0.08
    k_motor: float = 0.50
    decay: float = 0.01
    diffuse: float = 0.12
    k_noise: float = 0.00
    cap: float = 5.0
    boundary_speed: float = 0.04
    kernel: KernelCfg = field(default_factory=KernelCfg)
    env: FieldCfg = field(default_factory=FieldCfg)

def default_config() -> Dict:
    return asdict(Config())

def make_config_from_dict(d: Dict) -> Config:
    envd = d.get("env", {})
    kerd = d.get("kernel", {})
    return Config(
        seed=int(d.get("seed", 0)),
        frames=int(d.get("frames", 5000)),
        space=int(d.get("space", 64)),
        band=int(d.get("band", 3)),
        k_flux=float(d.get("k_flux", 0.08)),
        k_motor=float(d.get("k_motor", 0.50)),
        decay=float(d.get("decay", 0.01)),
        diffuse=float(d.get("diffuse", 0.12)),
        k_noise=float(d.get("k_noise", 0.0)),
        cap=float(d.get("cap", 5.0)),
        boundary_speed=float(d.get("boundary_speed", 0.04)),
        kernel=KernelCfg(
            enabled=bool(kerd.get("enabled", True)),
            radius=int(kerd.get("radius", 3)),
            lr=float(kerd.get("lr", 1e-3)),
            l2=float(kerd.get("l2", 1e-4)),
            init=float(kerd.get("init", 0.0)),
            k_gate=float(kerd.get("k_gate", 0.15)),
        ),
        env=FieldCfg(
            length=int(envd.get("length", 512)),
            frames=int(envd.get("frames", int(d.get("frames", 5000)))),
            noise_sigma=float(envd.get("noise_sigma", 0.0)),
            baseline=float(envd.get("baseline", 0.0)),
            amp_scale=float(envd.get("amp_scale", 1.0)),
            sources=envd.get("sources", FieldCfg().sources),
        ),
    )