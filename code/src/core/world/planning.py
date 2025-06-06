import logging
import copy
from typing import Dict, List, Tuple
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
        # start_pos = start_coords.get(agent_id)
        
        # subgoals = [start_pos]
        
        # if mission == 'get_snack':
        #     fridge_access_point = self.geometry.get_fridge_access_point()
        #     if not fridge_access_point:
        #         raise ValueError("Fridge access point not found for subgoals")
            
        #     door_node = self.world_graph.find_closest_door_to_agent(start_pos)
        #     if door_node is None:
        #         raise ValueError(f"No reachable door found for agent {agent_id}")
            
        #     subgoals.extend([fridge_access_point, door_node])
        
        # subgoals.append(start_pos)
        # return subgoals
    
        start_pos = start_coords.get(agent_id)
        fridge_pos = self.geometry.get_fridge_access_point()
        door_pos = self.world_graph.find_closest_door_to_agent(start_pos)

        # Convert coordinates to graph vertex IDs (VIDs)
        start_vid = self.world_graph.node_to_vid.get(start_pos)
        door_vid = self.world_graph.node_to_vid.get(door_pos)
        fridge_vid = self.world_graph.node_to_vid.get(fridge_pos)
        return_vid = start_vid

        return start_vid, door_vid, fridge_vid, return_vid


def compute_agent_path_sequences(agent_id: str,
                                 world_graph,
                                 geometry,
                                 start_coords: Dict[str, Tuple[int, int]],
                                 mission: str,
                                 max_steps: int) -> Tuple[List, List, List, List]:
    """
    Computes a unified set of path sequences for an agent based on a 4-segment journey:
    1. Start -> Door (shortest paths)
    2. Door -> Fridge (simple paths)
    3. Fridge -> Door (reverse of segment 2)
    4. Door -> Start (reverse of segment 1)

    Returns:
        A tuple containing four lists of paths, corresponding to the four segments.
    """
    logger = logging.getLogger(__name__)

    planner = SubgoalPlanner(world_graph, geometry)

    start, door, fridge, _ = planner.get_subgoals(agent_id, start_coords, mission)

    # Compute paths for each segment
    paths_start_to_door = get_shortest_paths(world_graph.igraph, start, door, world_graph.vid_to_node)
    logger.info(f"Agent {agent_id}: Found {len(paths_start_to_door)} shortest paths from Start to Door.")

    paths_door_to_fridge = get_simple_paths(world_graph.igraph, door, fridge, max_steps, world_graph.vid_to_node)
    logger.info(f"Agent {agent_id}: Found {len(paths_door_to_fridge)} simple paths from Door to Fridge (max_steps={max_steps}).")

    paths_fridge_to_door = [p[::-1] for p in paths_door_to_fridge]
    paths_door_to_start = [p[::-1] for p in paths_start_to_door]

    # Sort paths
    p1 = sorted(paths_start_to_door) if paths_start_to_door else []
    p2 = sorted(paths_door_to_fridge) if paths_door_to_fridge else []
    p3 = sorted(paths_fridge_to_door) if paths_fridge_to_door else []
    p4 = sorted(paths_door_to_start) if paths_door_to_start else []
    
    logger.info(
        f"Agent {agent_id} Path Segments: "
        f"Start --> Door={len(p1)}, Door --> Fridge={len(p2)}, "
        f"Fridge --> Door={len(p3)}, Door --> Start={len(p4)}"
    )

    return p1, p2, p3, p4