"""
Configuration validation schemas for simulation.

Provides validation schemas to ensure configuration parameters are valid.
"""

from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class SamplingSchema:
    """Schema for sampling configuration validation"""
    max_steps: int = 25
    max_steps_middle: int = 25
    sample_paths_suspect: int = 50
    sample_paths_detective: int = 1000
    seed: int = 42
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_steps_middle < 0:
            raise ValueError("max_steps_middle must be non-negative")
        if self.sample_paths_suspect <= 0:
            raise ValueError("sample_paths_suspect must be positive")
        if self.sample_paths_detective <= 0:
            raise ValueError("sample_paths_detective must be positive")


@dataclass 
class EvidenceSchema:
    """Schema for evidence configuration validation"""
    naive_detective_sigma: float = 1.0
    crumb_planting_sigma: float = 1.0
    sophisticated_detective_sigma: float = 1.0
    audio_similarity_sigma: float = 0.1
    audio_gt_step_size: int = 2
    visual_weight: float = 0.8
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not (0.0 <= self.visual_weight <= 1.0):
            raise ValueError("visual_weight must be between 0.0 and 1.0")
        if self.audio_similarity_sigma <= 0:
            raise ValueError("audio_similarity_sigma must be positive")
        if self.audio_gt_step_size <= 0:
            raise ValueError("audio_gt_step_size must be positive")


@dataclass
class AgentSchema:
    """Schema for agent configuration validation"""
    naive_temp: float = 0.01
    sophisticated_temp: float = 0.01
    weight: float = 0.1
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.naive_temp <= 0:
            raise ValueError("naive_temp must be positive")
        if self.sophisticated_temp <= 0:
            raise ValueError("sophisticated_temp must be positive")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("weight must be between 0.0 and 1.0")


@dataclass
class SimulationSchema:
    """Schema for simulation configuration validation"""
    door_close_prob: float = 0.0
    log_dir: str = "../../results"
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not (0.0 <= self.door_close_prob <= 1.0):
            raise ValueError("door_close_prob must be between 0.0 and 1.0")


def validate_evidence_type(evidence_type: str) -> str:
    """Validate evidence type parameter"""
    valid_types = ['visual', 'audio', 'multimodal']
    if evidence_type not in valid_types:
        raise ValueError(f"evidence_type must be one of {valid_types}, got {evidence_type}")
    return evidence_type


def validate_trial_name(trial_name: str) -> str:
    """Validate trial name parameter"""
    if not trial_name or not isinstance(trial_name, str):
        raise ValueError("trial_name must be a non-empty string")
    return trial_name 