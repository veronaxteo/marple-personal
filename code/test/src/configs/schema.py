from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml


@dataclass
class TrialConfig:
    """Configuration for trial file paths and settings"""
    name: str = "snack1"
    data_dir: Path = Path("trials")
    suspect_subdir: str = "suspect/json"
    detective_subdir: str = "detective/json"
    file_extension: str = ".json"
    
    def get_trial_paths(self) -> List[Path]:
        """Get list of trial file paths to search"""
        base_filename = f"{self.name}{self.file_extension}"
        return [
            self.data_dir / self.suspect_subdir / base_filename,
            self.data_dir / self.detective_subdir / base_filename,
            # Also try without subdirectories for flexibility
            self.data_dir / base_filename,
        ]


@dataclass
class SamplingConfig:
    """Path sampling parameters"""
    num_suspect_paths: int = 50
    num_detective_paths: int = 1000
    max_steps: int = 25
    max_steps_middle: int = 15
    
    # Temperature parameters for softmax
    naive_temp: float = 0.05
    sophisticated_temp: float = 0.05
    
    # Cost/utility parameters
    cost_weight: float = 0.5
    noisy_planting_sigma: float = 0.0


@dataclass
class EvidenceConfig:
    """Evidence processing configuration"""
    type: str = "visual"  # "visual", "audio", "multimodal"
    
    # Visual evidence parameters
    visual_smoothing_sigma: float = 0.0
    sophisticated_detective_sigma: float = 1.0
    sophisticated_suspect_sigma: float = 1.0
    
    # Audio evidence parameters  
    audio_similarity_sigma: float = 0.1
    audio_gt_step_size: int = 2
    
    # Multimodal parameters
    multimodal_weight: float = 0.5
    
    # Environment parameters
    door_close_prob: float = 0.0
    crumb_sigma: float = 0.0
    
    # Naive agent models (populated during simulation for sophisticated agents)
    naive_A_visual_likelihoods_map: Dict = field(default_factory=dict)
    naive_B_visual_likelihoods_map: Dict = field(default_factory=dict)
    naive_A_to_fridge_steps_model: List = field(default_factory=list)
    naive_A_from_fridge_steps_model: List = field(default_factory=list)
    naive_B_to_fridge_steps_model: List = field(default_factory=list)
    naive_B_from_fridge_steps_model: List = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Main simulation configuration"""
    # Meta information
    command: str = "rsm"
    seed: int = 42
    
    # Trial configuration
    trial: TrialConfig = field(default_factory=TrialConfig)
    
    # Logging and output
    log_dir: Path = Path("../../results")
    save_intermediate_results: bool = True
    
    # Core simulation components
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    
    # Empirical analysis (optional)
    empirical_paths: Optional[str] = None
    mismatched_analysis: bool = False

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimulationConfig":
        """Load configuration from .json or .yml/.yaml file."""
        path = Path(path)
        with path.open() as f:
            raw = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        return cls.from_dict(raw)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to .json or .yml/.yaml file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        
        # Convert Path objects to strings for serialization
        data = self._convert_paths_to_strings(data)
        
        with path.open("w") as f:
            if path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SimulationConfig":
        """Create config from dictionary"""
        raw = dict(raw)  # shallow copy
        
        # Handle nested configurations
        if "trial" in raw:
            if isinstance(raw["trial"], str):
                # Handle legacy format where trial was just a string
                raw["trial"] = {"name": raw["trial"]}
            raw["trial"] = TrialConfig(**raw["trial"])
        
        if "sampling" in raw:
            raw["sampling"] = SamplingConfig(**raw["sampling"])
        
        if "evidence" in raw:
            raw["evidence"] = EvidenceConfig(**raw["evidence"])
        
        # Convert string paths to Path objects
        if "log_dir" in raw:
            raw["log_dir"] = Path(raw["log_dir"])
            
        return cls(**raw)

    @classmethod
    def from_cli(
        cls,
        cfg_file: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
    ) -> "SimulationConfig":
        """Create configuration from file and CLI overrides.

        Args:
            cfg_file: Path to YAML/JSON config file
            overrides: List of "key=value" override strings
            
        Example:
            >>> cfg = SimulationConfig.from_cli(
            ...     cfg_file="configs/visual.yaml",
            ...     overrides=["sampling.num_detective_paths=500", "evidence.type=audio"]
            ... )
        """
        # Load base config from file or use defaults
        if cfg_file:
            base = cls.load(cfg_file)
        else:
            base = cls()
        
        # Apply CLI overrides
        if overrides:
            for override in overrides:
                key, value = override.split("=", 1)
                value = yaml.safe_load(value)  # Parse value (handles types automatically)
                
                # Navigate nested structure
                parts = key.split(".")
                obj = base
                for attr in parts[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, parts[-1], value)
        
        return base

    def _convert_paths_to_strings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert Path objects to strings for serialization"""
        if isinstance(data, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_paths_to_strings(item) for item in data]
        elif isinstance(data, Path):
            return str(data)
        else:
            return data

    @property
    def trial_name(self) -> str:
        """Get trial name for backward compatibility"""
        return self.trial.name
    
    @property
    def trial_paths(self) -> List[Path]:
        """Get list of trial file paths to search"""
        return self.trial.get_trial_paths()

