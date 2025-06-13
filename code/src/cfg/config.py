"""
Main configuration classes for simulation.

Provides dataclass-based configuration system with type safety and validation.
All default values are defined in YAML files, not in Python code.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SamplingConfig:
    """Configuration for path sampling behavior"""
    num_suspect_paths: int
    num_detective_paths: int
    max_steps: int
    seed: int
    naive_temp: float
    sophisticated_temp: float
    cost_weight: float
    

@dataclass
class EvidenceConfig:
    """Evidence-related configuration"""
    evidence_type: str
    naive_detective_sigma: float
    crumb_planting_sigma: float
    sophisticated_detective_sigma: float
    
    # Visual evidence settings
    visual_naive_likelihood_alpha: float
    visual_sophisticated_likelihood_alpha: float
    
    # Audio evidence settings (with defaults since they're optional)
    audio_similarity_sigma: float = 0.1
    audio_gt_step_size: int = 2
    
    # Naive agent models (for sophisticated agents) - always empty by default
    naive_A_visual_likelihoods_map: Dict = field(default_factory=dict)
    naive_B_visual_likelihoods_map: Dict = field(default_factory=dict)
    naive_A_to_fridge_steps_model: List = field(default_factory=list)
    naive_A_from_fridge_steps_model: List = field(default_factory=list)
    naive_B_to_fridge_steps_model: List = field(default_factory=list)
    naive_B_from_fridge_steps_model: List = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for agent behavior"""
    agent_id: str
    agent_type: str  # "naive", "sophisticated", "uniform"
    data_type: str 
    
    # Agent-specific models (populated during simulation)
    visual_likelihood_maps: Dict[str, Any] = field(default_factory=dict)
    audio_step_models: Dict[str, List] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Top-level simulation configuration"""
    trial_name: str
    seed: int
    
    # Component configurations
    sampling: SamplingConfig
    evidence: EvidenceConfig
    
    # Logging and output
    log_dir: str
    log_dir_base: str
    save_intermediate_results: bool = True 
    
    @classmethod
    def from_params(cls, base_config_path: str, **params) -> 'SimulationConfig':
        """
        Create configuration from YAML base and parameter overrides.
        
        This is the ONLY factory method - all other creation should use this.
        
        Args:
            base_config_path: Path to base YAML config (e.g., 'visual.yaml')
            **params: Parameters to override using dot notation keys
        
        Example:
            config = SimulationConfig.from_params(
                'visual.yaml',
                **{
                    'default_trial': 'snack2',
                    'sampling.naive_temp': 0.05,
                    'sampling.cost_weight': 10.0,
                    'evidence.naive_detective_sigma': 1.5,
                }
            )
        """
        # Load defaults first
        defaults_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
        with open(defaults_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        # Load custom config if provided and exists
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                custom = yaml.safe_load(f)
                defaults = cls._deep_update(defaults, custom)
        
        # Apply parameter overrides using dot notation
        if params:
            nested_overrides = cls._params_to_nested_dict(params)
            defaults = cls._deep_update(defaults, nested_overrides)
        
        # Extract components - all values must be present in YAML
        sampling_config = SamplingConfig(**defaults['sampling'])
        evidence_config = EvidenceConfig(**defaults['evidence'])
        
        return cls(
            trial_name=defaults['default_trial'],
            seed=defaults.get('seed', defaults['sampling']['seed']),  # Use sampling seed as fallback
            sampling=sampling_config,
            evidence=evidence_config,
            log_dir=defaults['simulation']['log_dir'],
            log_dir_base=defaults['simulation'].get('log_dir_base', defaults['simulation']['log_dir']),
            save_intermediate_results=defaults['simulation'].get('save_intermediate_results', True)
        )
    
    @staticmethod
    def _params_to_nested_dict(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dot notation parameters to nested dictionary"""
        result = {}
        for key, value in params.items():
            keys = key.split('.')
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return result
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep update dictionary, handling nested structures"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SimulationConfig._deep_update(result[key], value)
            else:
                result[key] = value
        return result


@dataclass
class DetectiveTaskConfig:
    """Configuration for a detective prediction task"""
    world: Any 
    trial_name: str
    sampled_data: Dict[str, Any]
    agent_type_being_simulated: str
    config: SimulationConfig
    
    # Optional parameters for empirical model
    source_data_type: Optional[str] = None
    mismatched_run: Optional[bool] = None
    param_log_dir: Optional[str] = None


@dataclass
class PathSamplingTask:
    """Configuration for a path sampling task"""
    world: Any
    agent_id: str
    simple_path_sequences: List
    config: SimulationConfig
    agent_type: str
    
    @property
    def num_sample_paths(self) -> int:
        """Get number of paths to sample based on context"""
        return self.config.sampling.num_suspect_paths
