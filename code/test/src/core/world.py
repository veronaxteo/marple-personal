import json
import logging
import os
import pickle
from pathlib import Path

# from src.configs.cfg import Simulationcfg
from src.configs import SimulationConfig
from src.core.paths import PathSampler
from src.utils.world_utils import WorldGraph, CoordinateMapper, WorldGeometry, SubgoalPlanner, compute_agent_path_sequences


class World:
    
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
        
        # Find kitchen info
        self.kitchen_info = next((r for r in info['rooms']['initial'] if r['type'] == 'Kitchen'), None)
        if self.kitchen_info is None:
            raise ValueError("Kitchen room information not found.")

        # Initialize components
        self._initialize_components(info)


    def _initialize_components(self, info):
        """Initialize all world components"""
        self.coordinate_mapper = CoordinateMapper(self.kitchen_info)
        self.geometry = WorldGeometry(info)
        self.world_graph = WorldGraph()
        self.world_graph.create_graph(info, self.geometry)
        self.subgoal_planner = SubgoalPlanner(self.world_graph, self.geometry)
        self.path_sampler = PathSampler()


    @staticmethod
    def initialize_world_start(trial_name_or_config):
        """Initialize World from trial configuration or name
        
        Args:
            trial_name_or_config: Either a trial name string (legacy) or SimulationConfig object
        """
        from src.configs import SimulationConfig
        from pathlib import Path
        
        # Handle both legacy string format and new config format
        if isinstance(trial_name_or_config, str):
            # Legacy format - trial name only
            trial_name = trial_name_or_config
            search_paths = [
                f'trials/suspect/json/{trial_name}_A1.json',
                f'trials/detective/json/{trial_name}_A1.json',
            ]
        else:
            # New format - use configuration object
            config = trial_name_or_config
            trial_name = config.trial_name
            base_filename = f"{trial_name}_A1.json"
            
            # Ensure data_dir is a Path object for proper path operations
            data_dir = Path(config.trial.data_dir)
            search_paths = [
                str(data_dir / config.trial.suspect_subdir / base_filename),
                str(data_dir / config.trial.detective_subdir / base_filename),
                str(data_dir / base_filename),  # Fallback
            ]

        logger = logging.getLogger(__name__)
        logger.debug(f"Searching for trial files: {search_paths}")
        
        json_path = None
        for path in search_paths:
            if os.path.exists(path):
                json_path = path
                logger.info(f"Found trial file: {path}")
                break
        
        if json_path is None:
            raise FileNotFoundError(f"Trial JSON file '{trial_name}' not found in any expected location:\n" + 
                                  "\n".join(f"  - {path}" for path in search_paths))

        with open(json_path, 'r') as f:
            data = json.load(f)
            trial_info = data.get('Grid')
            if trial_info is None: 
                raise KeyError(f"Expected 'Grid' key not found in {trial_name}")
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
        return self.geometry.get_fridge_access_point(self.world_graph.node_to_vid)

    def get_valid_kitchen_crumb_coords_world(self):
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

    # def get_subgoal_simple_path_sequences(self, agent_id: str, cfg: Simulationcfg, 
    def get_subgoal_simple_path_sequences(self, agent_id: str, cfg: SimulationConfig, 
                                        evidence_type_str: str, max_steps: int = 0):
        """Get path segments for given evidence_type_str type - delegates to utility function"""
        from src.utils.world_utils import compute_agent_path_sequences
        
        max_steps = max_steps if max_steps > 0 else cfg.sampling.max_steps
        
        return compute_agent_path_sequences(
            agent_id=agent_id,
            world_graph=self.world_graph,
            geometry=self.geometry,
            start_coords=self.start_coords,
            mission=self.mission,
            evidence_type_str=evidence_type_str,
            max_steps=max_steps
        )


def load_simple_path_sequences(log_dir_base: str, trial_name: str, w_t0: World, 
                             # cfg: Simulationcfg, 
                             cfg: SimulationConfig, 
                             max_steps: int):
    """Load or compute simple path sequences for a trial"""
    logger = logging.getLogger(__name__)
    
    # Determine cache file path
    cache_dir = os.path.join(log_dir_base, 'simple_paths')
    cache_file = f'{trial_name}_simple_paths_{cfg.evidence.type}_{cfg.sampling.max_steps}.pkl'
    cache_path = os.path.join(cache_dir, cache_file)

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded simple path sequences for {trial_name} from cache")
                return cached_data['A'], cached_data['B']
        except Exception as e:
            logger.warning(f"Failed to load cache for {trial_name}: {e}")
                    
    # Compute subgoal simple path sequences using utility function
    logger.info(f"Computing simple path sequences for {trial_name} (cfg.sampling.max_steps={cfg.sampling.max_steps})")
    
    paths_A = compute_agent_path_sequences('A', w_t0.world_graph, w_t0.geometry, 
                                         w_t0.start_coords, w_t0.mission, cfg.evidence.type, cfg.sampling.max_steps)
    paths_B = compute_agent_path_sequences('B', w_t0.world_graph, w_t0.geometry, 
                                         w_t0.start_coords, w_t0.mission, cfg.evidence.type, cfg.sampling.max_steps)
    
    # Cache results
    os.makedirs(cache_dir, exist_ok=True)
    try:
        cache_data = {'A': paths_A, 'B': paths_B}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cached simple path sequences for {trial_name}")
    except Exception as e:
        logger.error(f"Failed to cache paths for {trial_name}: {e}")

    return paths_A, paths_B

