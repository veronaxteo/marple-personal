"""
World module for handling world representation, navigation, and path planning.

This module provides:
- World: Main world class with component delegation
- WorldGraph: Graph representation and pathfinding
- WorldGeometry: Furniture and spatial queries  
- CoordinateMapper: Coordinate space transformations
- SubgoalPlanner: Path planning and subgoal generation
- load_or_compute_simple_path_sequences: Utility for cached path loading
"""

from .world import World
from .graph import WorldGraph, get_shortest_paths, get_simple_paths
from .geometry import WorldGeometry  
from .coordinates import CoordinateMapper
from .planning import SubgoalPlanner, compute_agent_path_sequences
from .utils import load_or_compute_simple_path_sequences

__all__ = [
    'World',
    'WorldGraph', 
    'WorldGeometry',
    'CoordinateMapper', 
    'SubgoalPlanner',
    'get_shortest_paths',
    'get_simple_paths', 
    'compute_agent_path_sequences',
    'load_or_compute_simple_path_sequences'
] 
