"""
Main configuration classes for simulation.

Provides dataclass-based configuration system with type safety and validation.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .schema import (
    SamplingSchema, EvidenceSchema, AgentSchema, SimulationSchema,
    validate_evidence_type, validate_trial_name
)


@dataclass
class SamplingConfig:
    """Configuration for path sampling behavior"""
    num_suspect_paths: int = 50
    num_detective_paths: int = 1000
    max_steps: int = 25
    max_steps_middle: int = 25
    seed: int = 42
    
    naive_temp: float = 0.01
    sophisticated_temp: float = 0.01
    
    cost_weight: float = 0.1
    
    def __post_init__(self):
        # Validate using schema
        SamplingSchema(
            max_steps=self.max_steps,
            max_steps_middle=self.max_steps_middle,
            sample_paths_suspect=self.num_suspect_paths,
            sample_paths_detective=self.num_detective_paths,
            seed=self.seed
        )


@dataclass
class EvidenceConfig:
    """Evidence-related configuration"""
    evidence_type: str = 'visual'
    
    naive_detective_sigma: float = 1.0 
    crumb_planting_sigma: float = 1.0 
    sophisticated_detective_sigma: float = 1.0
    
    # Audio evidence settings
    audio_similarity_sigma: float = 0.1
    audio_gt_step_size: int = 2
    
    # Multimodal evidence settings
    visual_weight: float = 0.5  # Weight for visual vs audio in multimodal
    
    # Naive agent models (for sophisticated agents)
    naive_A_visual_likelihoods_map: Dict = field(default_factory=dict)
    naive_B_visual_likelihoods_map: Dict = field(default_factory=dict)
    naive_A_to_fridge_steps_model: List = field(default_factory=list)
    naive_A_from_fridge_steps_model: List = field(default_factory=list)
    naive_B_to_fridge_steps_model: List = field(default_factory=list)
    naive_B_from_fridge_steps_model: List = field(default_factory=list)
    
    def __post_init__(self):
        # Validate evidence type and parameters
        self.evidence_type = validate_evidence_type(self.evidence_type)
        EvidenceSchema(
            audio_similarity_sigma=self.audio_similarity_sigma,
            audio_gt_step_size=self.audio_gt_step_size,
            visual_weight=self.visual_weight
        )


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
    seed: int = 42
    
    # Component configurations
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    
    # Logging and output
    log_dir: str = "../../results"
    log_dir_base: str = "../../results"  # Base directory for all logging
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        # Validate trial name
        self.trial_name = validate_trial_name(self.trial_name)
        
        # Validate simulation parameters
        SimulationSchema(log_dir=self.log_dir)
    
    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> 'SimulationConfig':
        """Create configuration from YAML file with optional overrides"""
        # Load defaults first
        defaults_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
        with open(defaults_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        # Load custom config if provided
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                custom = yaml.safe_load(f)
                defaults.update(custom)
        
        # Apply overrides
        defaults.update(overrides)
        
        # Extract components
        sampling_config = SamplingConfig(**defaults.get('sampling', {}))
        evidence_config = EvidenceConfig(**defaults.get('evidence', {}))
        
        return cls(
            trial_name=defaults.get('default_trial', 'snack1'),
            sampling=sampling_config,
            evidence=evidence_config,
            **defaults.get('simulation', {})
        )
    
    @classmethod
    def create_visual_config(
        cls, 
        trial_name: str, 
        cost_weight: float = 0.1, 
        naive_temp: float = 0.01,
        sophisticated_temp: float = 0.01,
        max_steps: int = 25,
        **kwargs
    ) -> 'SimulationConfig':
        """Factory method for visual evidence simulation"""
        sampling = SamplingConfig(
            cost_weight=cost_weight,
            naive_temp=naive_temp,
            sophisticated_temp=sophisticated_temp,
            max_steps=max_steps
        )
        evidence = EvidenceConfig(evidence_type="visual")
        
        return cls(
            trial_name=trial_name,
            sampling=sampling,
            evidence=evidence,
            **kwargs
        )
    
    @classmethod
    def create_audio_config(
        cls,
        trial_name: str,
        cost_weight: float = 0.1,
        naive_temp: float = 0.01,
        sophisticated_temp: float = 0.01,
        max_steps: int = 25,
        audio_similarity_sigma: float = 0.1,
        **kwargs
    ) -> 'SimulationConfig':
        """Factory method for audio evidence simulation"""
        sampling = SamplingConfig(
            cost_weight=cost_weight,
            naive_temp=naive_temp,
            sophisticated_temp=sophisticated_temp,
            max_steps=max_steps
        )
        evidence = EvidenceConfig(
            evidence_type="audio",
            audio_similarity_sigma=audio_similarity_sigma
        )
        
        return cls(
            trial_name=trial_name,
            sampling=sampling,
            evidence=evidence,
            **kwargs
        )
    
    @classmethod
    def create_multimodal_config(
        cls,
        trial_name: str,
        visual_weight: float = 0.5,
        **kwargs
    ) -> 'SimulationConfig':
        """Factory method for multimodal evidence simulation"""
        sampling = SamplingConfig(**kwargs)
        evidence = EvidenceConfig(
            evidence_type="multimodal",
            visual_weight=visual_weight
        )
        
        return cls(
            trial_name=trial_name,
            sampling=sampling,
            evidence=evidence,
            **kwargs
        )


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


def create_simulation_config(evidence_type: str = "visual", trial_name: str = "snack1", **kwargs) -> SimulationConfig:
    """Factory function to create simulation configuration"""
    if evidence_type == "visual":
        return SimulationConfig.create_visual_config(trial_name=trial_name, **kwargs)
    elif evidence_type == "audio":
        return SimulationConfig.create_audio_config(trial_name=trial_name, **kwargs)
    elif evidence_type == "multimodal":
        return SimulationConfig.create_multimodal_config(trial_name=trial_name, **kwargs)
    else:
        raise ValueError(f"Unknown evidence type: {evidence_type}") 