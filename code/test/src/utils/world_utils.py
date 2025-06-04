import logging
import os
import pickle
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from igraph import Graph
from .math_utils import furniture_size


@dataclass
class WorldInfo:
    """Container for world initialization data"""
    width: int
    height: int
    rooms: List[Dict]
    doors: List[Dict]
    agents: List[Dict]
    mission: str


class CoordinateMapper:
    """Handles coordinate transformations between world space and kitchen space"""
    
    def __init__(self, kitchen_info: Dict):
        self.kitchen_info = kitchen_info
        self.kitchen_width = kitchen_info['size'][0]
        self.kitchen_height = kitchen_info['size'][1] 
        self.kitchen_top_x = kitchen_info['top'][0]
        self.kitchen_top_y = kitchen_info['top'][1]
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def world_to_kitchen_coords(self, world_x: int, world_y: int) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to kitchen array coordinates"""
        if (self.kitchen_top_x <= world_x < self.kitchen_top_x + self.kitchen_width and
            self.kitchen_top_y <= world_y < self.kitchen_top_y + self.kitchen_height):
            return int(world_x - self.kitchen_top_x), int(world_y - self.kitchen_top_y)
        return None
    
    def kitchen_to_world_coords(self, kitchen_x: int, kitchen_y: int) -> Tuple[int, int]:
        """Convert kitchen array coordinates to world coordinates"""
        return int(kitchen_x + self.kitchen_top_x), int(kitchen_y + self.kitchen_top_y)


class WorldGeometry:
    """Handles furniture detection, door states, and spatial queries"""
    
    def __init__(self, world_info: Dict):
        self.world_info = world_info
        self.logger = logging.getLogger(self.__class__.__name__)
        self._valid_kitchen_crumb_coords_cache = None
    
    def is_furniture_at(self, location_tuple: Tuple[int, int]) -> bool:
        """Check if world coordinate is occupied by furniture (excluding crumbs)"""
        loc_x, loc_y = location_tuple
        for room in self.world_info['rooms']['initial']:
            for furniture in room['furnitures']['initial']:
                if furniture['type'] == 'crumbs':
                    continue
                f_x, f_y = furniture['pos']
                f_w, f_h = furniture_size.get(furniture['type'], (1, 1))
                if (f_x <= loc_x < f_x + f_w and f_y <= loc_y < f_y + f_h):
                    return True
        return False
    
    def get_initial_door_states(self) -> Dict[Tuple[int, int], str]:
        """Get initial door states as coordinate->state mapping"""
        return {tuple(door['pos']): door['state'] for door in self.world_info['doors']['initial']}
    
    def get_fridge_access_point(self, node_to_vid: Dict) -> Optional[Tuple[int, int]]:
        """Get fridge access point coordinate from world info"""
        kitchen_info = next((r for r in self.world_info['rooms']['initial'] if r['type'] == 'Kitchen'), None)
        fridge_info = next((f for f in kitchen_info['furnitures']['initial'] if f['type'] == 'electric_refrigerator'), None)
        fp = fridge_info['pos']
        fridge_access_point = (fp[0] - 1, fp[1] + 2)
        
        return fridge_access_point
    
    def get_valid_kitchen_crumb_coords(self, kitchen_info: Dict, node_to_vid: Dict, igraph: Graph) -> List[Tuple[int, int]]:
        """Get list of valid world coordinates for crumbs in kitchen"""
        if self._valid_kitchen_crumb_coords_cache is not None:
            return self._valid_kitchen_crumb_coords_cache
        
        valid_coords = []
        kx, ky = kitchen_info['top']
        kw, kh = kitchen_info['size']
        
        for world_y in range(ky, ky + kh):
            for world_x in range(kx, kx + kw):
                coord_tuple = (world_x, world_y)
                if coord_tuple in node_to_vid:
                    vid = node_to_vid[coord_tuple]
                    if not igraph.vs[vid]['is_door']:
                        valid_coords.append(coord_tuple)
        
        self._valid_kitchen_crumb_coords_cache = sorted(valid_coords)
        return self._valid_kitchen_crumb_coords_cache


class WorldGraph:
    """Handles graph creation, navigation, and pathfinding operations"""
    
    def __init__(self):
        self.igraph = None
        self.node_to_vid = {}
        self.vid_to_node = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_graph(self, world_info: Dict, geometry: WorldGeometry) -> None:
        """Create the igraph representation of the world"""
        self.igraph = Graph(directed=False)
        self.node_to_vid = {}
        self.vid_to_node = {}
        
        # Collect all nodes to add
        all_nodes_to_add = []
        edges_to_add = []
        
        # Add room nodes
        for room in world_info['rooms']['initial']:
            room_coords = []
            for x_coord in range(room['top'][0], room['top'][0] + room['size'][0]):
                for y_coord in range(room['top'][1], room['top'][1] + room['size'][1]):
                    loc = (x_coord, y_coord)
                    if not geometry.is_furniture_at(loc):
                        room_coords.append(loc)
                        if loc not in self.node_to_vid:
                            all_nodes_to_add.append({
                                'name': str(loc),
                                'coords': loc,
                                'is_door': False,
                                'room': room['type']
                            })
            
            # Add room edges
            for coord in room_coords:
                neighbor_h = (coord[0] + 1, coord[1])
                if neighbor_h in room_coords:
                    edges_to_add.append((coord, neighbor_h))
                neighbor_v = (coord[0], coord[1] + 1)
                if neighbor_v in room_coords:
                    edges_to_add.append((coord, neighbor_v))
        
        # Add door nodes
        for door_info in world_info['doors']['initial']:
            loc = tuple(door_info['pos'])
            if loc not in self.node_to_vid:
                all_nodes_to_add.append({
                    'name': str(loc),
                    'coords': loc,
                    'is_door': True,
                    'state': door_info['state'],
                    'room': None
                })
        
        # Batch add vertices
        self.igraph.add_vertices(len(all_nodes_to_add))
        for i, node_attrs in enumerate(all_nodes_to_add):
            self.igraph.vs[i]['name'] = node_attrs['name']
            self.igraph.vs[i]['coords'] = node_attrs['coords']
            self.igraph.vs[i]['is_door'] = node_attrs['is_door']
            self.igraph.vs[i]['room'] = node_attrs.get('room')
            if node_attrs['is_door']:
                self.igraph.vs[i]['state'] = node_attrs.get('state')
            
            self.node_to_vid[node_attrs['coords']] = i
            self.vid_to_node[i] = node_attrs['coords']
        
        # Batch add room edges
        igraph_edges = []
        for u_coord, v_coord in edges_to_add:
            if u_coord in self.node_to_vid and v_coord in self.node_to_vid:
                igraph_edges.append((self.node_to_vid[u_coord], self.node_to_vid[v_coord]))
        self.igraph.add_edges(igraph_edges)
        
        # Add door connection edges
        self._add_door_connections(world_info)
    
    def _add_door_connections(self, world_info: Dict) -> None:
        """Add edges connecting doors to adjacent rooms"""
        door_connection_edges = []
        for door_info in world_info['doors']['initial']:
            door_pos = tuple(door_info['pos'])
            potential_neighbors = []
            if door_info['dir'] == 'horz':
                potential_neighbors = [(door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1)]
            elif door_info['dir'] == 'vert':
                potential_neighbors = [(door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1])]
            
            if door_pos in self.node_to_vid:
                door_vid = self.node_to_vid[door_pos]
                for neighbor_pos in potential_neighbors:
                    if neighbor_pos in self.node_to_vid:
                        neighbor_vid = self.node_to_vid[neighbor_pos]
                        door_connection_edges.append(tuple(sorted((door_vid, neighbor_vid))))
        
        # Add unique door edges
        unique_door_edges = sorted(list(set(door_connection_edges)))
        self.igraph.add_edges(unique_door_edges)
    
    def find_closest_door_to_agent(self, agent_start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find closest door node to agent start position"""
        if agent_start_pos not in self.node_to_vid:
            self.logger.error(f"Agent start position {agent_start_pos} not found in graph")
            return None
        
        source_vid = self.node_to_vid[agent_start_pos]
        
        # Find all door vertices
        door_vids_and_pos = []
        for vid in range(len(self.igraph.vs)):
            vertex = self.igraph.vs[vid]
            try:
                if vertex['is_door'] is True and 'coords' in vertex.attributes():
                    door_vids_and_pos.append((vid, vertex['coords']))
            except KeyError:
                pass
        
        if not door_vids_and_pos:
            self.logger.warning("No valid door vertices found in graph")
            return None
        
        # Find closest door
        closest_door_pos = None
        min_dist = float('inf')
        
        for door_vid, door_pos_tuple in door_vids_and_pos:
            try:
                path_len_matrix = self.igraph.shortest_paths(source=source_vid, target=door_vid, weights=None, mode='all')
                if path_len_matrix and path_len_matrix[0]:
                    dist = path_len_matrix[0][0]
                    if dist != float('inf') and dist < min_dist:
                        min_dist = dist
                        closest_door_pos = door_pos_tuple
            except Exception as e:
                continue
        
        if closest_door_pos is None:
            self.logger.warning(f"No reachable door found from {agent_start_pos}")
        else:
            self.logger.debug(f"Closest door to {agent_start_pos} is {closest_door_pos} at distance {min_dist}")
        
        return closest_door_pos


class SubgoalPlanner:
    """Handles subgoal calculation and path segment generation"""
    
    def __init__(self, world_graph: WorldGraph, geometry: WorldGeometry):
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
            fridge_access_point = self.geometry.get_fridge_access_point(self.world_graph.node_to_vid)
            if not fridge_access_point:
                raise ValueError("Fridge access point not found for subgoals")
            
            door_node = self.world_graph.find_closest_door_to_agent(start_pos)
            if door_node is None:
                raise ValueError(f"No reachable door found for agent {agent_id}")
            
            subgoals.extend([fridge_access_point, door_node])
        
        subgoals.append(start_pos)
        return subgoals
    
    # TODO: might need to change depending how we compute simple paths for each cfg.evidence.type type
    def get_path_segments_visual(self, agent_id: str, subgoals: List[Tuple[int, int]], max_steps: int) -> Tuple[List, List, List, List]:
        """Get path segments for visual cfg.evidence.type: P1 (shortest), P2 (simple), P3 (shortest), P_FS (empty)"""
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
            sequences_p2 = get_simple_paths(self.world_graph.igraph, sg_vid_f, sg_vid_d, cfg.sampling.max_steps_middle, self.world_graph.vid_to_node)
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
        """Get path segments for audio cfg.evidence.type: P1 (simple), P_FS (simple, derived from P1), P2 & P3 (empty)"""
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


class PathSequenceCache:
    """Handles caching and loading of simple path sequences for World objects"""
    
    def __init__(self, log_dir_base: str):
        self.log_dir_base = log_dir_base
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # def load_simple_path_sequences(self, trial_name: str, world: 'World', cfg: 'Simulationcfg'):
    #     """Load or compute simple path sequences for a trial with caching."""
    #     cfg.sampling.max_steps = cfg.cfg.sampling.max_steps
    #     cfg.evidence.type_type_str = cfg.cfg.evidence.type

    #     cache_dir = os.path.join(self.log_dir_base, 'simple_paths')
    #     cache_filename = f'{trial_name}_simple_paths_{cfg.evidence.type_type_str}_{cfg.sampling.max_steps}.pkl'
    #     pickle_path = os.path.join(cache_dir, cache_filename)
        
    #     # Try to load from cache
    #     cached_result = self._load_from_cache(pickle_path, cfg.evidence.type_type_str, trial_name)
    #     if cached_result is not None:
    #         return cached_result
        
    #     # Compute new sequences
    #     self.logger.info(f"Computing {cfg.evidence.type_type_str} simple path sequences for {trial_name} (A and B)... Filename: {pickle_path}")
        
    #     # paths_A_tuple = self._compute_agent_paths(world, 'A', cfg, cfg.evidence.type_type_str, cfg.sampling.max_steps)
    #     # paths_B_tuple = self._compute_agent_paths(world, 'B', cfg, cfg.evidence.type_type_str, cfg.sampling.max_steps)
    #     paths_A_tuple = compute_agent_path_sequences('A', world.world_graph, world.geometry, 
    #                                      world.start_coords, world.mission, cfg.evidence.type_type_str, cfg.sampling.max_steps)
    #     paths_B_tuple = compute_agent_path_sequences('B', world.world_graph, world.geometry, 
    #                                      world.start_coords, world.mission, cfg.evidence.type_type_str, cfg.sampling.max_steps)
        
    #     # Cache results
    #     self._save_to_cache(pickle_path, paths_A_tuple, paths_B_tuple, cfg.evidence.type_type_str, trial_name)
    #     return paths_A_tuple, paths_B_tuple
    
    def _load_from_cache(self, pickle_path: str, evidence_type_str: str, trial_name: str):
        """Try to load cached path sequences."""
        if not os.path.exists(pickle_path):
            return None
            
        try:
            with open(pickle_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            if not (isinstance(cached_data, dict) and 'A' in cached_data and 'B' in cached_data):
                self.logger.warning(f"Cached {evidence_type_str} path data is not in expected dict format for {trial_name}. Recomputing.")
                return None
                
            paths_A_tuple = cached_data['A']
            paths_B_tuple = cached_data['B']
            
            # Validate 4-tuple structure
            if not (isinstance(paths_A_tuple, tuple) and len(paths_A_tuple) == 4 and
                   isinstance(paths_B_tuple, tuple) and len(paths_B_tuple) == 4):
                self.logger.warning(f"Cached {evidence_type_str} path tuples have incorrect structure for {trial_name}. Recomputing.")
                return None
                
            self.logger.info(f"Loaded cached {evidence_type_str} simple path sequences for {trial_name} from {pickle_path}.")
            
            # Log path counts for debugging
            if evidence_type_str == 'visual':
                self.logger.info(f"  A: P1:{len(paths_A_tuple[0])}, P2:{len(paths_A_tuple[1])}, P3:{len(paths_A_tuple[2])}")
                self.logger.info(f"  B: P1:{len(paths_B_tuple[0])}, P2:{len(paths_B_tuple[1])}, P3:{len(paths_B_tuple[2])}")
            elif evidence_type_str == 'audio':
                self.logger.info(f"  A: P1(S->F):{len(paths_A_tuple[0])}, P_FS(F->S):{len(paths_A_tuple[3])}")
                self.logger.info(f"  B: P1(S->F):{len(paths_B_tuple[0])}, P_FS(F->S):{len(paths_B_tuple[3])}")
                
            return paths_A_tuple, paths_B_tuple
            
        except Exception as e:
            self.logger.warning(f"Error loading cached {evidence_type_str} paths for {trial_name}: {e}. Recomputing.")
            return None
    
    
    def _save_to_cache(self, pickle_path: str, paths_A_tuple, paths_B_tuple, evidence_type_str: str, trial_name: str):
        """Save computed path sequences to cache."""
        try:
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            cached_data = {'A': paths_A_tuple, 'B': paths_B_tuple}
            with open(pickle_path, 'wb') as f:
                pickle.dump(cached_data, f)
            self.logger.info(f"Cached {evidence_type_str} simple path sequences for {trial_name} to {pickle_path}")
        except Exception as e:
            self.logger.error(f"Failed to cache {evidence_type_str} paths for {trial_name}: {e}")


def get_shortest_paths(igraph_instance, source_vid: int, target_vid: int, vid_to_node_map: dict):
    """Finds all shortest paths between source and target in an igraph graph."""
    logger = logging.getLogger(__name__)
    try:
        vid_paths = igraph_instance.get_all_shortest_paths(source_vid, to=target_vid, weights=None, mode='all')
        coord_paths = []
        for vid_path in vid_paths:
            coord_path = [vid_to_node_map[vid] for vid in vid_path]
            coord_paths.append(coord_path)
        return coord_paths
    except Exception as e:
        logger.warning(f"No path found between VIDs {source_vid} and {target_vid} using igraph, or error: {e}")
        return []


def get_simple_paths(igraph_instance, source_vid: int, target_vid: int, cutoff: int, vid_to_node_map: dict):
    """Finds all simple paths up to cutoff length using igraph."""
    logger = logging.getLogger(__name__)
    try:
        vid_paths = igraph_instance.get_all_simple_paths(source_vid, to=target_vid, cutoff=cutoff, mode='all')
        coord_paths = []
        for vid_path in vid_paths:
            coord_path = [vid_to_node_map[vid] for vid in vid_path]
            coord_paths.append(coord_path)
        return coord_paths
    except Exception as e:
        logger.warning(f"Error finding simple paths between vertex IDs {source_vid} and {target_vid} with cutoff {cutoff} using igraph: {e}")
        return []


def compute_agent_path_sequences(agent_id: str, world_graph: WorldGraph, geometry: WorldGeometry, 
                               start_coords: Dict[str, Tuple[int, int]], mission: str, 
                               evidence_type_str: str, max_steps: int) -> Tuple[List, List, List, List]:
    """
    Compute path sequences for an agent using world components directly.
    
    Args:
        agent_id: Agent identifier ('A' or 'B')
        world_graph: WorldGraph instance
        geometry: WorldGeometry instance
        start_coords: Dictionary mapping agent IDs to start coordinates
        mission: Mission string
        evidence_type_str: 'visual' or 'audio'  
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

    # Get path segments based on cfg.evidence.type type
    if evidence_type_str == 'visual':
        segments = subgoal_planner.get_path_segments_visual(agent_id, subgoals, max_steps)
    elif evidence_type_str == 'audio':
        segments = subgoal_planner.get_path_segments_audio(agent_id, subgoals, max_steps)
    else:
        logger.error(f"Unknown evidence_type_str type '{evidence_type_str}'")
        return [], [], [], []

    p1, p2, p3, p_fs = segments
    
    # Log path counts
    if evidence_type_str == 'visual':
        logger.info(f"Agent {agent_id} VISUAL: P1={len(p1)}, P2={len(p2)}, P3={len(p3)}")
    else:  # audio
        logger.info(f"Agent {agent_id} AUDIO: P1={len(p1)}, P_FS={len(p_fs)}")

    return p1, p2, p3, p_fs