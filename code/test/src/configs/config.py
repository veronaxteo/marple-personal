from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SamplingConfig:
    """Configuration for path sampling behavior"""
    num_suspect_paths: int = 50
    num_detective_paths: int = 1000
    max_steps: int = 25
    
    naive_temp: float = 0.01
    sophisticated_temp: float = 0.01
    
    w: float = 0.5
    noisy_planting_sigma: float = 0.0


@dataclass 
class EvidenceConfig:
    """Configuration for cfg.evidence.type processing"""
    type: str = "visual"  # "visual", "audio", "multimodal"
    
    # Visual cfg.evidence.type parameters
    visual_smoothing_sigma: float = 0.0
    sophisticated_detective_sigma: float = 0.0
    sophisticated_suspect_sigma: float = 0.0
    
    # Audio cfg.evidence.type parameters  
    audio_similarity_sigma: float = 0.1
    audio_gt_step_size: int = 2
    
    # Multimodal parameters
    w: float = 0.5  # For multimodal cfg.evidence.type
    
    # Naive agent models (for sophisticated agents)
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
    seed: int = 42
    
    # Component configurations
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    
    # Logging and output
    log_dir: str = "../../results"
    save_intermediate_results: bool = True
    
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
        """Factory method for visual cfg.evidence.type simulation"""
        sampling = SamplingConfig(
            cost_weight=cost_weight,
            naive_temp=naive_temp,
            sophisticated_temp=sophisticated_temp,
            max_steps=max_steps
        )
        evidence = EvidenceConfig(type="visual")
        
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
        """Factory method for audio cfg.evidence.type simulation"""
        sampling = SamplingConfig(
            cost_weight=cost_weight,
            naive_temp=naive_temp,
            sophisticated_temp=sophisticated_temp,
            max_steps=max_steps
        )
        evidence = EvidenceConfig(
            type="audio",
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
        w: float = 0.5,
        **kwargs
    ) -> 'SimulationConfig':
        """Factory method for multimodal cfg.evidence.type simulation"""
        sampling = SamplingConfig(**kwargs)
        evidence = EvidenceConfig(
            type="multimodal",
            w=w
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
        # This could be made smarter based on agent_type or other factors
        return self.config.sampling.num_suspect_paths
