"""
Base agent class for simulation agents.

Provides common functionality for suspects and detectives.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..cfg import SimulationConfig


class Agent(ABC):
    """Base class for all agents (suspects and detectives)"""
    
    def __init__(self, agent_id: str, data_type: str, config: SimulationConfig):
        self.agent_id = agent_id
        self.data_type = data_type
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{agent_id}]")
        self.logger.info(f"Initialized {self.__class__.__name__} with {data_type} data")
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process method to be implemented by subclasses"""
        pass 
