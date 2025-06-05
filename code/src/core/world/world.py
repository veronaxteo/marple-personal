import json
import logging
import os

from ...cfg import SimulationConfig
from ...core.paths import PathSampler
from .geometry import WorldGeometry
from .graph import WorldGraph
from .coordinates import CoordinateMapper
from .planning import SubgoalPlanner, compute_agent_path_sequences


class World:
    """
    This class handles the world representation, including the geometry, graph, and planning.
    """
    def __init__(self, info):
        self.info = info
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract basic world info
        self.width = info['width']
        self.height = info['height']
        self.mission = info['agents']['initial'][0]['cur_mission']
        self.start_coords = {
            'A': tuple(info['agents']['initial'][0]['pos']),
            'B': tuple(info['agents']['initial'][1]['pos'])
        }
        
        self.kitchen_info = next((r for r in info['rooms']['initial'] if r['type'] == 'Kitchen'), None)
        self._initialize_components(info)


    def _initialize_components(self, info):
        """Initialize all world components"""
        self.coordinate_mapper = CoordinateMapper(self.kitchen_info)
        self.geometry = WorldGeometry(info)
        self.world_graph = WorldGraph()
        self.world_graph.create_graph(info, self.geometry)
        self.subgoal_planner = SubgoalPlanner(self.world_graph, self.geometry)
        self.path_sampler = PathSampler(self)


    @staticmethod
    def initialize_world_start(filename):
        """Initialize World from trial JSON file"""
        # Try multiple potential paths for the trial file
        search_paths = [
            # From src/core/world/ go up three levels to code/, then to trials/
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'trials', 'suspect', 'json', filename),
            # From current working directory (should be code/)
            os.path.join('trials', 'suspect', 'json', filename),
            # Alternative relative paths
            f'../trials/suspect/json/{filename}',
            f'../../trials/suspect/json/{filename}'
        ]
        
        json_path = None
        for path in search_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                json_path = abs_path
                break
        
        if json_path is None:
            raise FileNotFoundError(f"Trial JSON file '{filename}' not found in any expected location. Searched: {[os.path.abspath(p) for p in search_paths]}")

        with open(json_path, 'r') as f:
            data = json.load(f)
            trial_info = data.get('Grid')
            if trial_info is None: 
                raise KeyError(f"Expected 'Grid' key not found in {filename}")
            return World(trial_info)

    # Coordinate mapping delegation
    def world_to_kitchen_coords(self, world_x, world_y):
        """Convert world coordinates to kitchen array coordinates"""
        return self.coordinate_mapper.world_to_kitchen_coords(world_x, world_y)

    def kitchen_to_world_coords(self, kitchen_x, kitchen_y):
        """Convert kitchen array coordinates to world coordinates"""
        return self.coordinate_mapper.kitchen_to_world_coords(kitchen_x, kitchen_y)

    # Geometry delegation
    def is_furniture_at(self, location_tuple):
        """Check if world coordinate is occupied by furniture"""
        return self.geometry.is_furniture_at(location_tuple)

    def get_initial_door_states(self):
        """Get initial door states as coordinate->state mapping"""
        return self.geometry.get_initial_door_states()

    def get_fridge_access_point(self):
        """Get fridge access point coordinate"""
        return self.geometry.get_fridge_access_point()

    def get_valid_kitchen_crumb_coords(self):
        """Get list of valid world coordinates for crumbs in kitchen"""
        return self.geometry.get_valid_kitchen_crumb_coords(self.kitchen_info, self.world_graph.node_to_vid, self.world_graph.igraph)

    # Graph operations
    def get_closest_door_to_agent(self, agent_id):
        """Find closest door node to agent start position"""
        agent_start_pos = self.start_coords.get(agent_id)
        if not agent_start_pos:
            self.logger.error(f"Start position not found for agent {agent_id}")
            return None
        return self.world_graph.find_closest_door_to_agent(agent_start_pos)

    # Subgoal and paths
    def get_subgoals(self, agent_id):
        """Get subgoal sequence for an agent"""
        return self.subgoal_planner.get_subgoals(agent_id, self.start_coords, self.mission)

    def get_subgoal_simple_path_sequences(self, agent_id: str, config: SimulationConfig, 
                                        evidence_type: str, max_steps: int = 0):
        """Get path segments for given evidence type - delegates to utility function"""
        
        max_steps = max_steps if max_steps > 0 else config.sampling.max_steps
        
        return compute_agent_path_sequences(
            agent_id=agent_id,
            world_graph=self.world_graph,
            geometry=self.geometry,
            start_coords=self.start_coords,
            mission=self.mission,
            evidence_type=evidence_type,
            max_steps=max_steps
        ) 
    