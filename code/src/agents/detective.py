"""
Detective agent implementation for evidence analysis and prediction generation.

Handles detective reasoning for both naive and sophisticated suspect models
using visual, audio, or multimodal evidence.
"""

import logging
from typing import Dict, Any

from .base import Agent
from ..cfg import SimulationConfig, DetectiveTaskConfig
from ..utils.evidence_utils import create_evidence_processor


class Detective(Agent):
    """
    Detective agent that analyzes evidence and generates predictions about suspect behavior.
    """
    
    def __init__(self, agent_id: str, config: SimulationConfig):
        super().__init__(agent_id, "detective", config)
    
    def simulate_detective(self, world, sampled_data: Dict[str, Any], agent_type_being_simulated: str, 
                          trial_name: str, param_log_dir: str) -> Dict[str, Any]:
        """
        Simulate detective reasoning and prediction generation.
        
        Args:
            world: World object containing environment information
            sampled_data: Dictionary containing sampled path data from suspects
            agent_type_being_simulated: 'naive' or 'sophisticated'
            trial_name: Name of the current trial
            param_log_dir: Directory for parameter logging
            
        Returns:
            Dictionary containing detective predictions and model outputs
        """
        self.logger.info(f"Simulating detective modeling {agent_type_being_simulated} suspects using {self.config.evidence.evidence_type} evidence")
        
        # Create detective task configuration
        task = DetectiveTaskConfig(
            world=world,
            sampled_data=sampled_data,
            agent_type_being_simulated=agent_type_being_simulated,
            trial_name=trial_name,
            param_log_dir=param_log_dir,
            config=self.config
        )
        
        # Create appropriate evidence processor
        evidence_processor = create_evidence_processor(self.config.evidence.evidence_type)
        
        # Compute predictions
        result = evidence_processor.compute_detective_predictions(task)
        
        return {
            'predictions': result.predictions,
            'model_output_A': result.model_output_A,
            'model_output_B': result.model_output_B,
            'prediction_data': result.prediction_data_for_json
        }
    
    def process(self, *args, **kwargs) -> Any:
        """Implementation of abstract method from base class"""
        return self.simulate_detective(*args, **kwargs) 
                