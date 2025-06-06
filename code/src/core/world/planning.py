import logging
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
    
    def get_subgoals(self, agent_id: str, start_coords: Dict[str, Tuple[int, int]]) -> Tuple[int, int, int]:
        """Get subgoals (start, door, fridge) as vertex IDs for an agent."""
        start_pos = start_coords.get(agent_id)
        fridge_pos = self.geometry.get_fridge_access_point()
        door_pos = self.world_graph.find_closest_door_to_agent(start_pos)

        start_vid = self.world_graph.node_to_vid.get(start_pos)
        door_vid = self.world_graph.node_to_vid.get(door_pos)
        fridge_vid = self.world_graph.node_to_vid.get(fridge_pos)
        
        return start_vid, door_vid, fridge_vid

def compute_agent_path_sequences(agent_id: str,
                                 world_graph,
                                 geometry,
                                 start_coords: Dict[str, Tuple[int, int]],
                                 max_steps: int) -> Tuple[List, List, List, List]:
    """
    Computes and returns the four path segments for an agent's journey.
    1. Start --> Door (shortest paths)
    2. Door --> Fridge (simple paths)
    3. Fridge --> Door (reverse of segment 2)
    4. Door --> Start (reverse of segment 1)
    """
    logger = logging.getLogger(__name__)
    planner = SubgoalPlanner(world_graph, geometry)
    start_vid, door_vid, fridge_vid = planner.get_subgoals(agent_id, start_coords)

    # Start --> Door
    paths_start_to_door = get_shortest_paths(world_graph.igraph, start_vid, door_vid, world_graph.vid_to_node)
    logger.info(f"Agent {agent_id}: Found {len(paths_start_to_door)} shortest paths from Start to Door.")

    # Door --> Fridge
    paths_door_to_fridge = get_simple_paths(world_graph.igraph, door_vid, fridge_vid, max_steps, world_graph.vid_to_node)
    logger.info(f"Agent {agent_id}: Found {len(paths_door_to_fridge)} simple paths from Door to Fridge (max_steps={max_steps}).")

    # Fridge --> Door
    paths_fridge_to_door = [p[::-1] for p in paths_door_to_fridge]
    
    # Door --> Start
    paths_door_to_start = [p[::-1] for p in paths_start_to_door]

    logger.info(
        f"Agent {agent_id} Base Path Segments: "
        f"Start --> Door = {len(paths_start_to_door)}, Door --> Fridge = {len(paths_door_to_fridge)}, "
        f"Fridge --> Door = {len(paths_fridge_to_door)}, Door --> Start = {len(paths_door_to_start)}"
    )

    return (
        sorted(paths_start_to_door),
        sorted(paths_door_to_fridge),
        sorted(paths_fridge_to_door),
        sorted(paths_door_to_start)
    )
