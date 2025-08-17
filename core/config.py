# core/config.py
from dataclasses import dataclass, asdict, field
from typing import List, Dict

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 1000
    noise_sigma: float = 0.01
    # sources: list of dicts, each like:
    # {"kind":"moving_peak","amp":1.0,"speed":0.10,"width":4.0,"start":24}
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed":  0.0, "width": 4.0, "start": 100}
        ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64
    k_flux: float = 0.05
    k_motor: float = 2.0
    diffuse: float = 0.05
    decay: float = 0.01
    env: FieldCfg = field(default_factory=FieldCfg)

def default_config() -> Dict:
    """Plain dict for Streamlit widgets/session_state defaults."""
    return asdict(Config())

def make_config_from_dict(d: Dict) -> Config:
    env_d = d.get("env", {})
    fcfg = FieldCfg(
        length=int(env_d.get("length", 512)),
        frames=int(env_d.get("frames", d.get("frames", 5000))),
        noise_sigma=float(env_d.get("noise_sigma", 0.01)),
        sources=env_d.get("sources", FieldCfg().sources),
    )
    return Config(
        seed=int(d.get("seed", 0)),
        frames=int(d.get("frames", 5000)),
        space=int(d.get("space", 64)),
        k_flux=float(d.get("k_flux", 0.05)),
        k_motor=float(d.get("k_motor", 2.0)),
        diffuse=float(d.get("diffuse", 0.05)),
        decay=float(d.get("decay", 0.01)),
        env=fcfg,
    )