"""
Configuration module for simulation settings.

Provides:
- SimulationConfig: Modern dataclass-based configuration system
- EvidenceConfig: Evidence-specific configuration
- SamplingConfig: Path sampling configuration  
- AgentConfig: Agent behavior configuration
- DetectiveTaskConfig: Task configuration for detective simulation
- PathSamplingTask: Task configuration for path sampling
- create_simulation_config: Factory function for creating configs
"""

from .config import (
    SimulationConfig,
    EvidenceConfig,
    SamplingConfig,
    AgentConfig,
    DetectiveTaskConfig,
    PathSamplingTask,
    create_simulation_config,
    validate_evidence_type,
    validate_trial_name
)

__all__ = [
    'SimulationConfig',
    'EvidenceConfig', 
    'SamplingConfig',
    'AgentConfig',
    'DetectiveTaskConfig',
    'PathSamplingTask',
    'create_simulation_config',
    
    # Validation functions
    'validate_evidence_type',
    'validate_trial_name'
] 