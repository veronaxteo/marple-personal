"""
Simulation module for different simulation strategies.

This module provides:
- BaseSimulator: Abstract base class for all simulators
- RSMSimulator: Rational Speech-act Model simulator
- EmpiricalSimulator: Empirical data analysis simulator  
- UniformSimulator: Uniform baseline model simulator
"""

from .base import BaseSimulator
from .rsm import RSMSimulator
from .empirical import EmpiricalSimulator
from .uniform import UniformSimulator

__all__ = [
    'BaseSimulator',
    'RSMSimulator',
    'EmpiricalSimulator', 
    'UniformSimulator'
] 