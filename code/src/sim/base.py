"""
Base simulator class for all simulation types.

Provides common functionality and error handling for all simulators.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..core.world import World
from ..cfg import SimulationConfig


class BaseSimulator(ABC):
    """
    Base simulator class that provides common functionality for all simulation types.
    """
    
    def __init__(self, config: SimulationConfig, log_dir_base: str, param_log_dir: str, trials_to_run: List[str]):
        self.config = config
        self.log_dir_base = log_dir_base
        self.param_log_dir = param_log_dir
        self.trials_to_run = trials_to_run
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Starting {self.__class__.__name__} simulation with {config.evidence.evidence_type} evidence")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run simulation for all specified trials.
        
        Returns:
            List of trial results
        """
        results = []
        
        for trial_file in self.trials_to_run:
            trial_name = trial_file.replace('_A1.json', '').replace('.json', '')
            
            try:
                self.logger.info(f"===== Running Trial: {trial_name} =====")
                
                # Initialize world
                world = World.initialize_world_start(trial_file)
                
                # Run trial-specific simulation
                trial_result = self.run_trial(trial_name, world)
                results.append(trial_result)
                
                self.logger.info(f"===== Finished Trial: {trial_name} =====")
                
            except Exception as e:
                self.logger.error(f"Error processing trial {trial_name}: {e}")
                # Log full traceback for debugging
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
        self.logger.info(f"{self.__class__.__name__} simulation completed")
        return results
    
    @abstractmethod
    def run_trial(self, trial_name: str, world: World) -> Dict[str, Any]:
        """
        Run simulation for a single trial.
        
        Args:
            trial_file: Filename of the trial JSON
            trial_name: Name of the trial (without extension)
            world: Initialized World object
            
        Returns:
            Dictionary containing trial results
        """
        pass 