from abc import ABC
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .math_utils import normalized_slider_prediction, smooth_likelihood_grid, smooth_likelihood_grid_connectivity_aware
from src.configs import SimulationConfig, EvidenceConfig, SamplingConfig
from src.evaluation.plot import plot_smoothing_comparison


@dataclass
class EvidenceTypeData:
    """Container for evidence-specific data"""
    visual_likelihoods_A: Optional[Dict] = None
    visual_likelihoods_B: Optional[Dict] = None
    audio_sequences_A: Optional[List] = None
    audio_sequences_B: Optional[List] = None
    coordinates: List[Tuple[int, int]] = None
    sequences: List[List] = None
    likelihoods: List[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class PredictionResult:
    """Container for detective prediction results"""
    predictions: Dict[str, Any]
    agent_A_model: Tuple[List, List]
    agent_B_model: Tuple[List, List]
    coordinates: List[Tuple[int, int]]


class EvidenceTypeProcessor(ABC):
    """Abstract base class for evidence type processors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_detective_predictions(self, task: SimulationConfig) -> PredictionResult:
        """Compute detective predictions using task configuration"""
        raise NotImplementedError


class VisualEvidenceTypeProcessor(EvidenceTypeProcessor):
    """Handles visual evidence processing and predictions"""
    
    def compute_detective_predictions(self, task: SimulationConfig) -> PredictionResult:
        """Compute visual evidence predictions using new config system"""
        
        self.logger.info(f"Computing VISUAL detective predictions for {task.agent_type_being_simulated} agents")
        
        # For now, return dummy data to avoid breaking the simulation
        # TODO: Implement proper visual evidence processing
        
        return PredictionResult(
            predictions={},
            agent_A_model=([], []),
            agent_B_model=([], []),
            coordinates=[]
        )


class AudioEvidenceTypeProcessor(EvidenceTypeProcessor):
    """Handles audio evidence processing and predictions"""
    
    def compute_detective_predictions(self, task: SimulationConfig) -> PredictionResult:
        """Compute audio evidence predictions using new config system"""
        
        self.logger.info(f"Computing AUDIO detective predictions for {task.agent_type_being_simulated} agents")
        
        # For now, return dummy data to avoid breaking the simulation
        # TODO: Implement proper audio evidence processing when needed
        
        return PredictionResult(
            predictions={},
            agent_A_model=([], []),
            agent_B_model=([], []),
            coordinates=[]
        )


class MultimodalEvidenceTypeProcessor(EvidenceTypeProcessor):
    """Handles multimodal evidence processing and predictions"""
    
    def compute_detective_predictions(self, task: SimulationConfig) -> PredictionResult:
        """Compute multimodal evidence predictions using new config system"""
        
        self.logger.info(f"Computing MULTIMODAL detective predictions for {task.agent_type_being_simulated} agents")
        
        # For now, return dummy data to avoid breaking the simulation
        # TODO: Implement proper multimodal evidence processing when needed
        
        return PredictionResult(
            predictions={},
            agent_A_model=([], []),
            agent_B_model=([], []),
            coordinates=[]
        )


def create_cfg(evidence_type: str) -> EvidenceTypeProcessor:
    """Factory function to create appropriate evidence type processor"""
    if evidence_type == 'visual':
        return VisualEvidenceTypeProcessor()
    elif evidence_type == 'audio':
        return AudioEvidenceTypeProcessor()
    elif evidence_type == 'multimodal':
        return MultimodalEvidenceTypeProcessor()
    else:
        raise ValueError(f"Unknown evidence type: {evidence_type}") 