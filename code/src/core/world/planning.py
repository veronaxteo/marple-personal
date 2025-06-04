import logging
import copy
from typing import Dict, List, Tuple, Optional
from .graph import get_shortest_paths, get_simple_paths


class SubgoalPlanner:
    """
    This class handles the subgoal calculation and path segment generation.
    """
    
    def __init__(self, world_graph, geometry):
        self.world_graph = world_graph
        self.geometry = geometry
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_subgoals(self, agent_id: str, start_coords: Dict[str, Tuple[int, int]], mission: str) -> List[Tuple[int, int]]:
        """Get subgoal sequence for an agent"""
        start_pos = start_coords.get(agent_id)
        if not start_pos:
            raise ValueError(f"Start position not found for agent {agent_id}")
        
        subgoals = [start_pos]
        
        if mission == 'get_snack':
            fridge_access_point = self.geometry.get_fridge_access_point()
            if not fridge_access_point:
                raise ValueError("Fridge access point not found for subgoals")
            
            door_node = self.world_graph.find_closest_door_to_agent(start_pos)
            if door_node is None:
                raise ValueError(f"No reachable door found for agent {agent_id}")
            
            subgoals.extend([fridge_access_point, door_node])
        
        subgoals.append(start_pos)
        return subgoals
    
    def get_path_segments_visual(self, agent_id: str, subgoals: List[Tuple[int, int]], max_steps_middle: int) -> Tuple[List, List, List, List]:
        """Get path segments for visual evidence: P1 (shortest), P2 (simple), P3 (shortest), P_FS (empty)"""
        if len(subgoals) < 4:
            self.logger.error(f"Not enough subgoals for agent {agent_id} (expected 4, got {len(subgoals)})")
            return [], [], [], []
        
        # Convert coordinates to VIDs
        subgoal_vids = []
        for sg_coord in subgoals:
            if sg_coord in self.world_graph.node_to_vid:
                subgoal_vids.append(self.world_graph.node_to_vid[sg_coord])
            else:
                self.logger.error(f"Subgoal coordinate {sg_coord} not found in graph for agent {agent_id}")
                return [], [], [], []
        
        sg_vid_s0, sg_vid_f, sg_vid_d, sg_vid_s1 = subgoal_vids[0], subgoal_vids[1], subgoal_vids[2], subgoal_vids[3]
        
        try:
            sequences_p1 = get_shortest_paths(self.world_graph.igraph, sg_vid_s0, sg_vid_f, self.world_graph.vid_to_node)
            sequences_p2 = get_simple_paths(self.world_graph.igraph, sg_vid_f, sg_vid_d, max_steps_middle, self.world_graph.vid_to_node)
            sequences_p3 = get_shortest_paths(self.world_graph.igraph, sg_vid_d, sg_vid_s1, self.world_graph.vid_to_node)
            sequences_fridge_to_start = []
            
            return (sorted(sequences_p1) if sequences_p1 else [],
                   sorted(sequences_p2) if sequences_p2 else [],
                   sorted(sequences_p3) if sequences_p3 else [],
                   sequences_fridge_to_start)
        except Exception as e:
            self.logger.error(f"Error during visual path segment computation for agent {agent_id}: {e}")
            return [], [], [], []
    

    def get_path_segments_audio(self, agent_id: str, subgoals: List[Tuple[int, int]], max_steps: int) -> Tuple[List, List, List, List]:
        """Get path segments for audio evidence: P1 (simple), P_FS (simple, derived from P1), P2 & P3 (empty)"""
        if len(subgoals) < 2:
            self.logger.error(f"Not enough subgoals for agent {agent_id} for audio (expected at least 2, got {len(subgoals)})")
            return [], [], [], []
        
        # Convert coordinates to VIDs
        subgoal_vids = []
        for sg_coord in subgoals:
            if sg_coord in self.world_graph.node_to_vid:
                subgoal_vids.append(self.world_graph.node_to_vid[sg_coord])
            else:
                self.logger.error(f"Subgoal coordinate {sg_coord} not found in graph for agent {agent_id}")
                return [], [], [], []
        
        sg_vid_s0, sg_vid_f = subgoal_vids[0], subgoal_vids[1]
        
        try:
            # P1: Start -> Fridge (simple paths)
            p1_paths = get_simple_paths(self.world_graph.igraph, sg_vid_s0, sg_vid_f, max_steps, self.world_graph.vid_to_node)
            sequences_p1 = sorted(p1_paths) if p1_paths else []
            
            # P_FS: Fridge -> Start (reverse P1 paths)
            sequences_fridge_to_start = []
            if sequences_p1:
                for path in sequences_p1:
                    reversed_path = copy.deepcopy(path)
                    reversed_path.reverse()
                    if (len(reversed_path) - 1) <= max_steps:
                        sequences_fridge_to_start.append(reversed_path)
                sequences_fridge_to_start = sorted(sequences_fridge_to_start)
            
            sequences_p2 = []
            sequences_p3 = []
            
            return sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start
        except Exception as e:
            self.logger.error(f"Error during audio path segment computation for agent {agent_id}: {e}")
            return [], [], [], []


def compute_agent_path_sequences(agent_id: str, world_graph, geometry, 
                               start_coords: Dict[str, Tuple[int, int]], mission: str, 
                               evidence_type: str, max_steps: int) -> Tuple[List, List, List, List]:
    """
    Compute path sequences for an agent using world components directly.
    
    Args:
        agent_id: Agent identifier ('A' or 'B')
        world_graph: WorldGraph instance
        geometry: WorldGeometry instance
        start_coords: Dictionary mapping agent IDs to start coordinates
        mission: Mission string
        evidence_type: 'visual' or 'audio'
        max_steps: Maximum steps for pathfinding
        
    Returns:
        Tuple of (P1, P2, P3, P_FS) path lists
    """
    logger = logging.getLogger(__name__)
    
    # Create subgoal planner
    subgoal_planner = SubgoalPlanner(world_graph, geometry)
    
    try:
        subgoals = subgoal_planner.get_subgoals(agent_id, start_coords, mission)
    except ValueError as e:
        logger.error(f"Failed to get subgoals for agent {agent_id}: {e}")
        return [], [], [], []

    if len(subgoals) < 4:
        logger.error(f"Insufficient subgoals for agent {agent_id} (expected 4, got {len(subgoals)})")
        return [], [], [], []

    # Get path segments based on evidence type
    if evidence_type == 'visual':
        segments = subgoal_planner.get_path_segments_visual(agent_id, subgoals, max_steps)
    elif evidence_type == 'audio':
        segments = subgoal_planner.get_path_segments_audio(agent_id, subgoals, max_steps)
    else:
        logger.error(f"Unknown evidence type '{evidence_type}'")
        return [], [], [], []

    p1, p2, p3, p_fs = segments
    
    # Log path counts
    if evidence_type == 'visual':
        logger.info(f"Agent {agent_id} VISUAL: P1={len(p1)}, P2={len(p2)}, P3={len(p3)}")
    else:  # audio
        logger.info(f"Agent {agent_id} AUDIO: P1={len(p1)}, P_FS={len(p_fs)}")

    return p1, p2, p3, p_fs 