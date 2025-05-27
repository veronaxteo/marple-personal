from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class SimulationParams:
    command: str
    trial: str
    log_dir: str
    max_steps: int
    w: Optional[float] = None
    n_temp: Optional[float] = None
    s_temp: Optional[float] = None 
    paths: Optional[str] = None 
    mismatched: Optional[bool] = False 

    door_close_prob: float = 0.0
    soph_suspect_sigma: float = 1.0
    soph_detective_sigma: float = 1.0
    noisy_planting_sigma: float = 0.5
    
    naive_A_crumb_likelihoods_map: Optional[Dict[Any, Any]] = field(default_factory=dict)
    naive_B_crumb_likelihoods_map: Optional[Dict[Any, Any]] = field(default_factory=dict)

    sample_paths_suspect: int = 50 
    sample_paths_detective: int = 1000

    seed: int = 42
    