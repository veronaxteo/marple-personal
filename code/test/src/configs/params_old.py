from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class Simulationcfg:
    command: str
    trial: str
    evidence: str = 'visual'
    log_dir: Optional[str] = '../../results'
    max_steps: Optional[int] = 25
    w: Optional[float] = 0.5
    sampling.naive_temp: Optional[float] = 0.05
    sampling.sophisticated_temp: Optional[float] = 0.05 

    door_close_prob: float = 0.0
    soph_suspect_sigma: float = 1.0
    soph_detective_sigma: float = 3.0
    noisy_planting_sigma: float = 1.0
    
    audio_gt_step_size: int = 2

    naive_A_visual_likelihoods_map: Optional[Dict[tuple, float]] = field(default_factory=dict)
    naive_B_visual_likelihoods_map: Optional[Dict[tuple, float]] = field(default_factory=dict)
    
    naive_A_to_fridge_steps_model: List[int] = field(default_factory=list)
    naive_A_from_fridge_steps_model: List[int] = field(default_factory=list)
    naive_B_to_fridge_steps_model: List[int] = field(default_factory=list)
    naive_B_from_fridge_steps_model: List[int] = field(default_factory=list)

    cfg.evidence.audio_similarity_sigma: float = 0.1 

    cfg.sampling.num_suspect_paths: int = 50 
    cfg.sampling.num_suspect_paths: int = 1000
    seed: int = 42

    # empirical
    paths: Optional[str] = None 
    mismatched: Optional[bool] = False

