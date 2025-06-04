"""
Suspect agent implementation for path generation and sampling.

Handles both naive and sophisticated suspect behaviors with different
sampling strategies based on evidence type.
"""

import logging
from typing import Dict, Any

from .base import Agent
from ..cfg import SimulationConfig, PathSamplingTask


class Suspect(Agent):
    """
    Suspect agent that generates path samples based on agent behavior type.
    """
    
    def __init__(self, agent_id: str, config: SimulationConfig):
        super().__init__(agent_id, "suspect", config)
    
    def simulate_suspect(self, world, paths_A, paths_B, agent_type: str, num_suspect_paths: int) -> Dict[str, Any]:
        """
        Simulate suspect agent path generation.
        
        Args:
            world: World object containing environment information
            paths_A: Path sequences for agent A  
            paths_B: Path sequences for agent B
            agent_type: 'naive' or 'sophisticated'
            num_suspect_paths: Number of paths to generate per agent
            
        Returns:
            Dictionary containing sampled path data for both agents
        """
        self.logger.info(f"Simulating {agent_type} suspect paths using {self.config.evidence.evidence_type} evidence")
        
        # Create sampling tasks for both agents
        task_A = PathSamplingTask(
            world=world,
            agent_id='A',
            agent_type=agent_type,
            simple_path_sequences=paths_A,
            config=self.config
        )
        
        task_B = PathSamplingTask(
            world=world,
            agent_id='B', 
            agent_type=agent_type,
            simple_path_sequences=paths_B,
            config=self.config
        )
        
        # Sample paths for both agents
        agent_A_data = world.path_sampler.sample_paths(task_A)
        agent_B_data = world.path_sampler.sample_paths(task_B)
        
        return {
            'A': agent_A_data,
            'B': agent_B_data
        }
    
    def process(self, *args, **kwargs) -> Any:
        """Implementation of abstract method from base class"""
        return self.simulate_suspect(*args, **kwargs) 
    