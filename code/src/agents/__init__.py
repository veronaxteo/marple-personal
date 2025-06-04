"""
Agent module for simulation agents.

This module provides:
- Agent: Base agent class with common functionality
- Suspect: Suspect agent that generates paths based on utility functions  
- Detective: Detective agent that makes predictions about suspects based on evidence
"""

from .base import Agent
from .suspect import Suspect
from .detective import Detective

__all__ = [
    'Agent',
    'Suspect', 
    'Detective'
] 