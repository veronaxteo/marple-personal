import json
import networkx as nx
# import igraph as ig # igraph import will be added later if needed directly in this file, or handled in utils
from igraph import Graph # Directly import Graph for now
import numpy as np
import logging
import os
import copy
from joblib import Parallel, delayed
from test_utils import softmax_list_vals, normalized_slider_prediction, get_shortest_paths, get_simple_paths
from globals import furniture_size
from params import SimulationParams
from test_evidence import get_compressed_audio_from_path, get_segmented_audio_likelihood, single_segment_audio_likelihood
import pickle


class World(object):
    """
    World representation:
    - Uses path sequences (lists of world tuples) internally for logic.
    - Generates numbered 2D arrays (kitchen coords) for output/saving.
    - Calculates likelihood only based on the middle sequence, i.e. path from fridge --> door.
    """
    def __init__(self, info):
        self.width = 0
        self.height = 0
        self.info = info
        # self.graph = None # No longer needed
        self.igraph = None
        self.node_to_vid = {}
        self.vid_to_node = {}
        self.kitchen_info = None
        self.kitchen_width = 0
        self.kitchen_height = 0
        self.kitchen_top_x = 0
        self.kitchen_top_y = 0
        self.start_coords = {}
        self.mission = None
        self._valid_kitchen_crumb_coords_world = None
        self.audio_framing_cache = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.create_world()


    def create_world(self):
        """Builds the NetworkX graph and stores key info."""
        self.width = self.info['width']
        self.height = self.info['height']
        self.mission = self.info['agents']['initial'][0]['cur_mission']

        self.igraph = Graph(directed=False)
        self.node_to_vid = {}
        self.vid_to_node = {}
        current_vid = 0

        self.kitchen_info = next((r for r in self.info['rooms']['initial'] if r['type'] == 'Kitchen'), None)
        if self.kitchen_info is None:
            raise ValueError("Kitchen room information not found.")

        self.kitchen_width = self.kitchen_info['size'][0]
        self.kitchen_height = self.kitchen_info['size'][1]
        self.kitchen_top_x = self.kitchen_info['top'][0]
        self.kitchen_top_y = self.kitchen_info['top'][1]

        self.start_coords = {
            'A': tuple(self.info['agents']['initial'][0]['pos']),
            'B': tuple(self.info['agents']['initial'][1]['pos'])
        }

        # Create graph using igraph
        all_nodes_to_add = [] # List of (name_str, attributes_dict)
        edges_to_add = []

        for r in self.info['rooms']['initial']:
            room_coords = []
            for x_coord in range(r['top'][0], r['top'][0] + r['size'][0]):
                for y_coord in range(r['top'][1], r['top'][1] + r['size'][1]):
                    loc = (x_coord, y_coord)
                    if not self.is_furniture_at(loc):
                        room_coords.append(loc)
                        if loc not in self.node_to_vid:
                            all_nodes_to_add.append({
                                'name': str(loc), 
                                'coords': loc, 
                                'is_door': False, 
                                'room': r['type']
                            })
                            # Actual adding to graph and map will be done after collecting all nodes
            
            # Add edges for this room (will be processed after all nodes are in graph)
            for coord in room_coords:
                # Check horizontal neighbor
                neighbor_h = (coord[0] + 1, coord[1])
                if neighbor_h in room_coords:
                    edges_to_add.append((coord, neighbor_h))
                # Check vertical neighbor (grid_2d_graph from networkx adds this way)
                neighbor_v = (coord[0], coord[1] + 1)
                if neighbor_v in room_coords:
                    edges_to_add.append((coord, neighbor_v))

        # Add door nodes to the list of nodes to add
        for d_info in self.info['doors']['initial']:
            loc = tuple(d_info['pos'])
            if loc not in self.node_to_vid:
                 all_nodes_to_add.append({
                    'name': str(loc),
                    'coords': loc,
                    'is_door': True,
                    'state': d_info['state'],
                    'room': None # Doors are not in rooms technically
                })
        
        # Batch add vertices
        self.igraph.add_vertices(len(all_nodes_to_add))
        for i, node_attrs in enumerate(all_nodes_to_add):
            self.igraph.vs[i]['name'] = node_attrs['name']
            self.igraph.vs[i]['coords'] = node_attrs['coords']
            self.igraph.vs[i]['is_door'] = node_attrs['is_door']
            self.igraph.vs[i]['room'] = node_attrs.get('room') # Handles doors not having room
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
        door_connection_edges = []
        for d_info in self.info['doors']['initial']:
            door_pos = tuple(d_info['pos'])
            potential_neighbors = []
            if d_info['dir'] == 'horz':
                potential_neighbors = [(door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1)]
            elif d_info['dir'] == 'vert':
                potential_neighbors = [(door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1])]
            
            if door_pos in self.node_to_vid:
                door_vid = self.node_to_vid[door_pos]
                for neighbor_pos in potential_neighbors:
                    if neighbor_pos in self.node_to_vid: # Check if neighbor is a valid node (part of a room, not furniture)
                        neighbor_vid = self.node_to_vid[neighbor_pos]
                        # Avoid adding duplicate edges if any were already added by room logic (unlikely for doors)
                        door_connection_edges.append(tuple(sorted((door_vid, neighbor_vid))))
        
        # Add door connection edges, ensuring no duplicates from this step either
        unique_door_edges = sorted(list(set(door_connection_edges)))
        self.igraph.add_edges(unique_door_edges)
        
        # Original NetworkX graph creation (commented out)
        # # Create graph
        # for r in self.info['rooms']['initial']:
        #     room_graph = nx.grid_2d_graph(
        #         range(r['top'][0], r['top'][0] + r['size'][0]),
        #         range(r['top'][1], r['top'][1] + r['size'][1]),
        #         create_using=nx.Graph
        #     )
        #     nodes_to_remove = {loc for loc in room_graph.nodes if self.is_furniture_at(loc)}
        #     room_graph.remove_nodes_from(nodes_to_remove)
        #     for node in room_graph.nodes:
        #          room_graph.nodes[node]['is_door'] = False
        #          room_graph.nodes[node]['room'] = r['type']
        #     self.graph = nx.compose(self.graph, room_graph)

        # for d in self.info['doors']['initial']:
        #     door_pos = tuple(d['pos'])
        #     self.graph.add_node(door_pos, is_door=True, state=d['state'], room=None)
        #     potential_neighbors = []
        #     if d['dir'] == 'horz':
        #         potential_neighbors = [(door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1)]
        #     elif d['dir'] == 'vert':
        #         potential_neighbors = [(door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1])]
        #     for neighbor_pos in potential_neighbors:
        #         if self.graph.has_node(neighbor_pos):
        #             self.graph.add_edge(door_pos, neighbor_pos)


    @staticmethod
    def initialize_world_start(filename):
        """Initializes World from trial JSON."""
        json_path = f'../trials/suspect/json/{filename}'
        if not os.path.exists(json_path):
            alt_json_path = os.path.join(os.path.dirname(__file__), '..', '.._trials', 'suspect', 'json', filename)
            if not os.path.exists(alt_json_path):
                final_alt_path = os.path.join('trials','suspect','json', filename)
                if not os.path.exists(final_alt_path):
                    raise FileNotFoundError(f"Trial JSON file not found at {json_path}, {alt_json_path}, or {final_alt_path}")
                else:
                    json_path = final_alt_path    
            else:
                json_path = alt_json_path

        with open(json_path, 'r') as f:
            trial_info = json.load(f).get('Grid')
            if trial_info is None: raise KeyError("Expected 'Grid' key not found.")
            return World(trial_info)


    def is_furniture_at(self, location_tuple):
        """Checks if world coord tuple is occupied by furniture (excluding crumbs)."""
        loc_x, loc_y = location_tuple
        for r in self.info['rooms']['initial']:
            for f in r['furnitures']['initial']:
                if f['type'] == 'crumbs': continue
                f_x, f_y = f['pos']
                f_w, f_h = furniture_size.get(f['type'], (1, 1))
                if (f_x <= loc_x < f_x + f_w and f_y <= loc_y < f_y + f_h):
                    return True
        return False


    def get_initial_door_states(self):
        """Returns a dictionary of initial door states {door_pos_tuple: state_string}."""
        return {tuple(d['pos']): d['state'] for d in self.info['doors']['initial']}


    def get_fridge_access_point(self):
        """Returns the world coordinate tuple for the fridge access point."""
        if self.mission != 'get_snack': return None
        fridge_info = next((f for f in self.kitchen_info['furnitures']['initial'] if f['type'] == 'electric_refrigerator'), None)
        
        if fridge_info is None: raise ValueError("Refrigerator not found for access point calculation.")
        fp = fridge_info['pos']
        fridge_access_point = (fp[0] - 1, fp[1] + 2)
        # if not self.graph.has_node(fridge_access_point): # NX version
        if fridge_access_point not in self.node_to_vid: # IG version: check if coord is a known node
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    alt_access_point = (fp[0] + dx, fp[1] + dy)
                    # if self.graph.has_node(alt_access_point): # NX version
                    if alt_access_point in self.node_to_vid: # IG version
                        logging.warning(f"Default fridge access point {fridge_access_point} invalid. Using {alt_access_point} instead.")
                        return alt_access_point
            raise ValueError(f"Calculated Fridge access point {fridge_access_point} and its neighbors are not valid nodes in the graph.")
        return fridge_access_point


    def world_to_kitchen_coords(self, world_x, world_y):
        """Convert world coordinates (wx, wy) to kitchen array coordinates (kx, ky)."""
        if (self.kitchen_top_x <= world_x < self.kitchen_top_x + self.kitchen_width and
            self.kitchen_top_y <= world_y < self.kitchen_top_y + self.kitchen_height):
            # kx = col index, ky = row index
            return int(world_x - self.kitchen_top_x), int(world_y - self.kitchen_top_y)
        return None 


    def kitchen_to_world_coords(self, kitchen_x, kitchen_y):
        """Convert kitchen array coords (kx, ky) to world coords (wx, wy)."""
        return int(kitchen_x + self.kitchen_top_x), int(kitchen_y + self.kitchen_top_y)


    def get_valid_kitchen_crumb_coords_world(self):
        """Returns list of valid world coord tuples (wx, wy) for crumbs in kitchen."""
        if self._valid_kitchen_crumb_coords_world is not None:
            return self._valid_kitchen_crumb_coords_world
        
        valid_coords = []
        kx, ky = self.kitchen_info['top']
        kw, kh = self.kitchen_info['size']
        
        for world_y in range(ky, ky + kh):
            for world_x in range(kx, kx + kw):
                coord_tuple = (world_x, world_y)
                # if self.graph.has_node(coord_tuple) and not self.graph.nodes[coord_tuple].get('is_door', False): # NX version
                if coord_tuple in self.node_to_vid: # IG version: check if coord is a known node
                    vid = self.node_to_vid[coord_tuple]
                    if not self.igraph.vs[vid]['is_door']:
                        valid_coords.append(coord_tuple)
        self._valid_kitchen_crumb_coords_world = sorted(valid_coords)
        
        return self._valid_kitchen_crumb_coords_world


    def get_closest_door_to_agent(self, agent_id):
        """Finds closest door node to agent start (world coords)."""
        self.logger.debug(f"Finding closest door for agent {agent_id}")
        agent_start_pos = self.start_coords.get(agent_id)
        if not agent_start_pos or agent_start_pos not in self.node_to_vid:
            self.logger.error(f"Agent start position {agent_start_pos} for agent {agent_id} not found in node_to_vid map.")
            return None
        source_vid = self.node_to_vid[agent_start_pos]
        
        door_vids_and_pos = []
        # Iterate through all vertices in the igraph graph
        for vid in range(len(self.igraph.vs)):
            vertex = self.igraph.vs[vid]
            try:
                # Check if 'is_door' attribute exists and is True
                if vertex['is_door'] is True:
                    # Check if 'coords' attribute exists
                    if 'coords' in vertex.attributes():
                        door_vids_and_pos.append((vid, vertex['coords']))
                    else:
                        self.logger.warning(f"Door vertex VID {vid} is missing 'coords' attribute.")
            except KeyError as e:
                # This can happen if a vertex somehow doesn't have 'is_door' attribute
                # Or if it's None and direct access fails (though direct access is preferred over .get for igraph.Vertex)
                # self.logger.debug(f"Vertex VID {vid} missing attribute {e} while checking for doors.")
                pass # Simply skip if it's not a properly formed door vertex

        if not door_vids_and_pos:
            self.logger.warning(f"No valid door VIDs found in igraph for agent {agent_id}.")
            return None

        closest_door_pos_coord = None
        min_dist = float('inf')
        
        for door_vid, original_door_pos_tuple in door_vids_and_pos:
            try:
                path_len_matrix = self.igraph.shortest_paths(source=source_vid, target=door_vid, weights=None, mode='all')
                if path_len_matrix and path_len_matrix[0]:
                    dist = path_len_matrix[0][0]
                    if dist != float('inf') and dist < min_dist:
                        min_dist = dist
                        closest_door_pos_coord = original_door_pos_tuple
            except Exception as e:
                # self.logger.debug(f"No path or error finding path from agent {agent_id} (VID {source_vid}) to door VID {door_vid}: {e}")
                continue
        
        if closest_door_pos_coord is None:
            self.logger.warning(f"Agent {agent_id} could not find a reachable door.")
        else:
            self.logger.debug(f"Closest door for agent {agent_id} is {closest_door_pos_coord} at distance {min_dist}.")
        return closest_door_pos_coord


    def get_subgoals(self, agent_id):
        """Get subgoals sequence (world coords tuples) for an agent."""
        start_pos = self.start_coords.get(agent_id)

        if not start_pos: raise ValueError(f"Start pos not found for agent {agent_id}")
        subgoals = [start_pos]

        if self.mission == 'get_snack':
            fridge_access_point = self.get_fridge_access_point()
            if not fridge_access_point: raise ValueError("Fridge access point not found for subgoals.")
            
            door_node = self.get_closest_door_to_agent(agent_id)
            if door_node is None: raise ValueError(f"No reachable door found for agent {agent_id} for subgoals.")
            subgoals.extend([fridge_access_point, door_node])
        
        subgoals.append(start_pos)
        return subgoals


    def _get_visual_path_segments(self, agent_id: str, subgoals: list, max_steps_middle: int) -> tuple:
        # Helper for visual evidence: P1 (shortest), P2 (simple), P3 (shortest), P_FS (empty)
        logger = logging.getLogger(__name__)
        
        # Convert subgoal coordinates to VIDs
        subgoal_vids = []
        for sg_coord in subgoals:
            if sg_coord in self.node_to_vid:
                subgoal_vids.append(self.node_to_vid[sg_coord])
            else:
                logger.error(f"Subgoal coordinate {sg_coord} not found in node_to_vid map for agent {agent_id}. Cannot compute visual paths.")
                return [], [], [], []
        
        if len(subgoal_vids) < 4: # Expected S, F, D, S
            logger.error(f"Not enough valid subgoal VIDs for agent {agent_id} (expected 4, got {len(subgoal_vids)}).")
            return [], [], [], []

        sg_vid_s0, sg_vid_f, sg_vid_d, sg_vid_s1 = subgoal_vids[0], subgoal_vids[1], subgoal_vids[2], subgoal_vids[3]

        try:
            n_jobs = 8
            print(f"Computing visual path segments with n_jobs={n_jobs}")
            results = Parallel(n_jobs=n_jobs)(
                [
                    delayed(get_shortest_paths)(self.igraph, sg_vid_s0, sg_vid_f, self.vid_to_node),  # P1: Start -> Fridge (shortest)
                    delayed(get_simple_paths)(self.igraph, sg_vid_f, sg_vid_d, max_steps_middle, self.vid_to_node),  # P2: Fridge -> Door (simple)
                    delayed(get_shortest_paths)(self.igraph, sg_vid_d, sg_vid_s1, self.vid_to_node)  # P3: Door -> Start (shortest)
                ]
            )
            sequences_p1 = sorted(results[0]) if results[0] is not None else []
            sequences_p2 = sorted(results[1]) if results[1] is not None else []
            sequences_p3 = sorted(results[2]) if results[2] is not None else []
            sequences_fridge_to_start = []  # P_FS is not used for visual
        except Exception as e:
            logger.error(f"Error during visual path segment computation (igraph) for agent {agent_id}: {e}", exc_info=True)
            return [], [], [], []
        return sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start

    def _get_audio_path_segments(self, agent_id: str, subgoals: list, params: SimulationParams) -> tuple:
        # Helper for audio evidence: P1 (simple), P_FS (simple, derived from P1 by reversing), P2 & P3 (empty)
        logger = logging.getLogger(__name__)
        sequences_p1 = []
        sequences_fridge_to_start = []

        # Convert subgoal coordinates to VIDs
        subgoal_vids = []
        for sg_coord in subgoals:
            if sg_coord in self.node_to_vid:
                subgoal_vids.append(self.node_to_vid[sg_coord])
            else:
                logger.error(f"Subgoal coordinate {sg_coord} not found in node_to_vid map for agent {agent_id}. Cannot compute audio paths.")
                return [], [], [], []
        
        if len(subgoal_vids) < 2: # Expected at least S, F for P1
            logger.error(f"Not enough valid subgoal VIDs for agent {agent_id} for P1 (expected at least 2, got {len(subgoal_vids)}).")
            return [], [], [], []
        
        sg_vid_s0, sg_vid_f = subgoal_vids[0], subgoal_vids[1]

        try:
            # P1: Start -> Fridge (simple paths)
            p1_paths = get_simple_paths(self.igraph, sg_vid_s0, sg_vid_f, params.max_steps, self.vid_to_node)
            sequences_p1 = sorted(p1_paths) if p1_paths is not None else []

            # P_FS: Fridge -> Start (simple paths) - by reversing P1 paths
            # TODO: faster way to reverse paths?
            if sequences_p1:
                for path in sequences_p1: 
                    reversed_path = copy.deepcopy(path)
                    reversed_path.reverse() 
                    if (len(reversed_path) -1) <= params.max_steps:
                        sequences_fridge_to_start.append(reversed_path)
                sequences_fridge_to_start = sorted(sequences_fridge_to_start)
            
            sequences_p2 = [] 
            sequences_p3 = []

        except Exception as e:
            logger.error(f"Error during audio path segment computation (igraph) for agent {agent_id}: {e}", exc_info=True)
            return [], [], [], [] 
        return sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start

    def get_subgoal_simple_path_sequences(self, agent_id: str, params: SimulationParams, evidence_type: str, max_steps_middle: int = 0):
        # max_steps_middle is only used for visual, provide a default
        logger = logging.getLogger(__name__)
        try:
            subgoals = self.get_subgoals(agent_id)
        except ValueError as e:
            logger.error(f"Failed to get subgoals for agent {agent_id} in trial (world hash: {hash(str(self.info))}): {e}")
            return [], [], [], []

        if len(subgoals) < 4: # Start, Fridge, Door, Start
            logger.error(f"Subgoal generation resulted in too few subgoals for agent {agent_id} (expected 4, got {len(subgoals)}): {subgoals}")
            return [], [], [], []

        if evidence_type == 'visual':
            if max_steps_middle <= 0:
                max_steps_middle = params.max_steps # Default if not provided for visual
            sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = self._get_visual_path_segments(agent_id, subgoals, max_steps_middle)
        elif evidence_type == 'audio':
            sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = self._get_audio_path_segments(agent_id, subgoals, params)
        else:
            logger.error(f"Unknown evidence type '{evidence_type}' for path sequence generation.")
            return [], [], [], []

        # Log counts for the primary paths used by each modality
        if evidence_type == 'visual':
            logger.info(f"Agent {agent_id} VISUAL paths: P1(S->F):{len(sequences_p1)}, P2(F->D):{len(sequences_p2)}, P3(D->S):{len(sequences_p3)}")
        elif evidence_type == 'audio':
            logger.info(f"Agent {agent_id} AUDIO paths: P1(S->F):{len(sequences_p1)}, P_FS(F->S):{len(sequences_fridge_to_start)}")

        if not sequences_p1 and (evidence_type == 'audio' or evidence_type == 'visual'):
             logger.warning(f"Agent {agent_id} ({evidence_type}): P1 (Start->Fridge) sequences are empty. Subgoals: {subgoals}")
        if evidence_type == 'audio' and not sequences_fridge_to_start:
             logger.warning(f"Agent {agent_id} ({evidence_type}): P_FS (Fridge->Start) sequences are empty. Subgoals: {subgoals}")
        if evidence_type == 'visual' and not sequences_p2:
            logger.info(f"Agent {agent_id} ({evidence_type}): P2 (Fridge->Door) sequences are empty. Subgoals: {subgoals}")
            
        return sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start


    def get_sample_paths(
        self, 
        agent_id: str, 
        simple_path_sequences: list, 
        num_sample_paths: int, 
        params: SimulationParams,
        agent_type: str
    ):
        logger = logging.getLogger(__name__)
        # simple_path_sequences is now (P1, P2, P3, P_FS)
        # P1: Start -> Fridge
        # P2: Fridge -> Door
        # P3: Door -> Start
        # P_FS: Fridge -> Start (new, for audio from_fridge)
        sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = simple_path_sequences

        sampled_full_sequences = []
        sampled_middle_sequences = [] # For audio, this will store P_FS (Fridge->Start)
        sampled_to_fridge_sequences = [] # For audio, this will store P1 (Start->Fridge)
        sampled_chosen_plant_spots = [] # Visual only
        sampled_audio_sequences_compressed = [] # For audio, compressed signature of P1+P_FS
        sampled_full_sequence_lengths = []
        sampled_to_fridge_sequence_lengths = []
        sampled_middle_sequence_lengths = [] # For audio, length of P_FS

        if params.evidence == 'visual':
            if not all([sequences_p1, sequences_p2, sequences_p3]):
                logger.warning(f"Agent {agent_id} ({agent_type}) VISUAL: P1, P2, or P3 segments empty. Cannot sample paths.")
                return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

            path_utilities = np.ones(len(sequences_p2)) # Visual utility is based on P2 (Fridge->Door)
            middle_path_lengths = np.array([len(seq) for seq in sequences_p2])
            min_len, max_len = np.min(middle_path_lengths), np.max(middle_path_lengths)
            rescaled_lengths = np.zeros_like(middle_path_lengths, dtype=float)
            if max_len > min_len: rescaled_lengths = (middle_path_lengths - min_len) / (max_len - min_len)

            current_path_utilities = []
            path_optimal_plant_spots_for_p2 = [None] * len(sequences_p2)

            if agent_type == 'sophisticated':
                fridge_access_point = self.get_fridge_access_point()
                for idx, p2_seq in enumerate(sequences_p2):
                    optimal_plant_spot_for_this_p2_seq, best_achievable_slider_for_path = self._calculate_optimal_plant_spot_and_slider(
                        agent_id, p2_seq, fridge_access_point, params.naive_A_visual_likelihoods_map, params.naive_B_visual_likelihoods_map
                    )
                    path_optimal_plant_spots_for_p2[idx] = optimal_plant_spot_for_this_p2_seq
                    path_framing_metric_scaled = best_achievable_slider_for_path / 100.0
                    cost_factor = params.w * rescaled_lengths[idx]
                    utility_factor = params.w * (1 - rescaled_lengths[idx])
                    if agent_id == 'A': current_path_utilities.append(utility_factor + (1 - params.w) * path_framing_metric_scaled)
                    else: current_path_utilities.append(utility_factor - (1 - params.w) * path_framing_metric_scaled)
            elif agent_type == 'naive' or agent_type == 'uniform': # Uniform also uses cost for visual
                for l_rescaled in rescaled_lengths:
                    current_path_utilities.append(params.w * (1 - l_rescaled))
            path_utilities = np.array(current_path_utilities) if current_path_utilities else np.ones(len(sequences_p2))
            
            if len(path_utilities) > 0 and not np.all(path_utilities == path_utilities[0]):
                probabilities = softmax_list_vals(path_utilities, temp=params.n_temp if agent_type == 'naive' else params.s_temp)
            else:
                probabilities = np.full(len(sequences_p2), 1.0 / len(sequences_p2)) if len(sequences_p2) > 0 else []

            if not probabilities.size and len(sequences_p2) > 0 : # Handles case where len(sequences_p2)=0 above
                 logger.warning(f"VISUAL Probabilities array is empty for agent {agent_id} ({agent_type}) but sequences_p2 is not. Defaulting to uniform.")
                 probabilities = np.full(len(sequences_p2), 1.0 / len(sequences_p2))
            elif not probabilities.size and not sequences_p2:
                logger.warning(f"VISUAL No P2 sequences available for agent {agent_id} ({agent_type}). Cannot sample paths.")
                # Return empty as before
                return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

            num_first = len(sequences_p1)
            num_middle = len(sequences_p2) # This is P2 (Fridge->Door)
            num_last = len(sequences_p3)

            for _ in range(num_sample_paths):
                idx1 = np.random.randint(0, num_first)
                idx3 = np.random.randint(0, num_last)
                idx2 = np.random.choice(num_middle, p=probabilities) # P2 is chosen based on utility

                p1_seq = sequences_p1[idx1]       # Start -> Fridge
                p2_seq = sequences_p2[idx2]       # Fridge -> Door
                p3_seq = sequences_p3[idx3]       # Door -> Start
                
                full_sequence = p1_seq[:-1] + p2_seq[:-1] + p3_seq
                sampled_full_sequences.append(full_sequence)
                sampled_middle_sequences.append(p2_seq) # For visual, middle is P2 (Fridge->Door)
                sampled_to_fridge_sequences.append(p1_seq)
                sampled_full_sequence_lengths.append(len(full_sequence) -1 if full_sequence else 0)
                sampled_to_fridge_sequence_lengths.append(len(p1_seq) -1 if p1_seq else 0)
                sampled_middle_sequence_lengths.append(len(p2_seq) -1 if p2_seq else 0)

                chosen_plant_spot_for_sample = None
                if agent_type == 'sophisticated' and path_optimal_plant_spots_for_p2[idx2] is not None:
                    optimal_spot = path_optimal_plant_spots_for_p2[idx2]
                    if params.noisy_planting_sigma > 0:
                        chosen_plant_spot_for_sample = self._get_noisy_plant_spot(optimal_spot, params.noisy_planting_sigma)
                    else:
                        chosen_plant_spot_for_sample = optimal_spot
                sampled_chosen_plant_spots.append(chosen_plant_spot_for_sample)

        elif params.evidence == 'audio':
            logger.info(f"AUDIO path sampling for agent {agent_id} ({agent_type})")
            candidate_paths_to_fridge = sequences_p1
            candidate_paths_from_fridge = sequences_fridge_to_start # Fridge -> Start

            if not candidate_paths_to_fridge or not candidate_paths_from_fridge:
                logger.warning(f"Agent {agent_id} ({agent_type}) AUDIO: Missing To_Fridge or From_Fridge_To_Start paths. Cannot sample.")
                return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

            logger.info(f"Agent {agent_id} ({agent_type}) AUDIO: Num raw candidate_paths_to_fridge: {len(candidate_paths_to_fridge)}")
            logger.info(f"Agent {agent_id} ({agent_type}) AUDIO: Num raw candidate_paths_from_fridge: {len(candidate_paths_from_fridge)}")
            
            # Step 1: Group paths by length
            paths_by_len_to = {}
            for p_seq in candidate_paths_to_fridge:
                p_len = len(p_seq) - 1 if p_seq else 0
                if p_len < 0: p_len = 0 # Ensure non-negative length
                if p_len not in paths_by_len_to:
                    paths_by_len_to[p_len] = []
                paths_by_len_to[p_len].append(p_seq)

            paths_by_len_from = {}
            for p_seq in candidate_paths_from_fridge:
                p_len = len(p_seq) - 1 if p_seq else 0
                if p_len < 0: p_len = 0 # Ensure non-negative length
                if p_len not in paths_by_len_from:
                    paths_by_len_from[p_len] = []
                paths_by_len_from[p_len].append(p_seq)

            if not paths_by_len_to or not paths_by_len_from:
                logger.warning(f"Agent {agent_id} ({agent_type}) AUDIO: No valid length groups for to/from paths. Cannot sample.")
                return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

            # Step 2: Create length-pair metadata
            length_pair_metadata_list = []
            unique_lengths_to = sorted(list(paths_by_len_to.keys()))
            unique_lengths_from = sorted(list(paths_by_len_from.keys()))

            for eval_steps_to in unique_lengths_to:
                for eval_steps_from in unique_lengths_from:
                    # Ensure there are actually paths for these lengths before adding
                    if eval_steps_to in paths_by_len_to and eval_steps_from in paths_by_len_from:
                        cost = eval_steps_to + eval_steps_from
                        length_pair_metadata_list.append({
                            'eval_steps_to': eval_steps_to,
                            'eval_steps_from': eval_steps_from,
                            'cost': cost
                        })
            
            logger.info(f"Agent {agent_id} ({agent_type}) AUDIO: Total unique length-pair combinations for utility calculation: {len(length_pair_metadata_list)}")

            if not length_pair_metadata_list:
                logger.warning(f"Agent {agent_id} ({agent_type}) AUDIO: No valid length pair metadata created.")
                return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

            costs = np.array([data['cost'] for data in length_pair_metadata_list])
            rescaled_costs = np.zeros_like(costs, dtype=float)
            min_cost, max_cost = np.min(costs), np.max(costs)
            if max_cost > min_cost: rescaled_costs = (costs - min_cost) / (max_cost - min_cost)
            
            audio_length_pair_utilities = [] # Renamed from audio_pair_utilities
            sigma_factor = params.audio_segment_similarity_sigma

            for i, length_meta in enumerate(length_pair_metadata_list):
                eval_steps_to = length_meta['eval_steps_to']
                eval_steps_from = length_meta['eval_steps_from']
                current_rescaled_cost = rescaled_costs[i]
                utility = 0.0
                if agent_type == 'naive' or agent_type == 'uniform':
                    utility = params.w * (1 - current_rescaled_cost)
                elif agent_type == 'sophisticated':
                    cache_key = (eval_steps_to, eval_steps_from)
                    if cache_key in self.audio_framing_cache: 
                        L_A_total, L_B_total = self.audio_framing_cache[cache_key] 
                    else:
                        # Ensure models are not empty before calculating mean, default to 0 likelihood if no model steps
                        model_A_to_steps = params.naive_A_to_fridge_steps_model
                        model_A_from_steps = params.naive_A_from_fridge_steps_model
                        model_B_to_steps = params.naive_B_to_fridge_steps_model
                        model_B_from_steps = params.naive_B_from_fridge_steps_model

                        L_A_to = np.mean([single_segment_audio_likelihood(model_s_to, eval_steps_to, sigma_factor) for model_s_to in model_A_to_steps]) if model_A_to_steps else 0
                        L_A_from = np.mean([single_segment_audio_likelihood(model_s_from, eval_steps_from, sigma_factor) for model_s_from in model_A_from_steps]) if model_A_from_steps else 0
                        L_A_total = L_A_to * L_A_from
                        
                        L_B_to = np.mean([single_segment_audio_likelihood(model_s_to, eval_steps_to, sigma_factor) for model_s_to in model_B_to_steps]) if model_B_to_steps else 0
                        L_B_from = np.mean([single_segment_audio_likelihood(model_s_from, eval_steps_from, sigma_factor) for model_s_from in model_B_from_steps]) if model_B_from_steps else 0
                        L_B_total = L_B_to * L_B_from
                        self.audio_framing_cache[cache_key] = (L_A_total, L_B_total) 
                    
                    predicted_slider = normalized_slider_prediction(L_A_total, L_B_total)
                    path_framing_metric_scaled = predicted_slider / 100.0
                    cost_utility_comp = params.w * (1 - current_rescaled_cost)
                    framing_utility_comp = (1 - params.w) * path_framing_metric_scaled
                    if agent_id == 'A': utility = cost_utility_comp + framing_utility_comp
                    else: utility = cost_utility_comp - framing_utility_comp
                audio_length_pair_utilities.append(utility)

            if not audio_length_pair_utilities:
                logger.warning(f"Agent {agent_id} ({agent_type}) AUDIO: No utilities calculated for length pairs. Defaulting to uniform if length pairs exist.")
                probabilities = np.full(len(length_pair_metadata_list), 1.0 / len(length_pair_metadata_list)) if length_pair_metadata_list else []
            elif np.all(np.array(audio_length_pair_utilities) == audio_length_pair_utilities[0]):
                probabilities = np.full(len(length_pair_metadata_list), 1.0 / len(length_pair_metadata_list))
            else:
                probabilities = softmax_list_vals(np.array(audio_length_pair_utilities), temp=params.n_temp if agent_type == 'naive' else params.s_temp)

            if not probabilities.size and length_pair_metadata_list:
                 logger.warning(f"AUDIO Probabilities array is empty for agent {agent_id} ({agent_type}) but length_pair_metadata_list exists. Defaulting to uniform.")
                 probabilities = np.full(len(length_pair_metadata_list), 1.0 / len(length_pair_metadata_list))
            elif not length_pair_metadata_list: 
                logger.warning(f"AUDIO No length_pair_metadata_list available for agent {agent_id} ({agent_type}). Cannot sample paths.")
                return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

            for _ in range(num_sample_paths):
                chosen_idx = np.random.choice(len(length_pair_metadata_list), p=probabilities)
                chosen_length_meta = length_pair_metadata_list[chosen_idx]
                
                selected_len_to = chosen_length_meta['eval_steps_to']
                selected_len_from = chosen_length_meta['eval_steps_from']

                # Randomly pick actual paths that match the chosen lengths
                # Ensure that the lists of paths for the selected lengths are not empty
                possible_paths_to = paths_by_len_to.get(selected_len_to)
                possible_paths_from = paths_by_len_from.get(selected_len_from)

                if not possible_paths_to or not possible_paths_from:
                    logger.error(f"Critical Error: No paths available for chosen length pair ({selected_len_to}, {selected_len_from}). This should not happen. Skipping sample.")
                    # Potentially append None or handle error more gracefully, e.g., by re-sampling or logging problematic length pair
                    sampled_audio_sequences_compressed.append(None)
                    sampled_full_sequences.append([]) # Or appropriate empty/error value
                    sampled_to_fridge_sequences.append([])
                    sampled_middle_sequences.append([])
                    sampled_full_sequence_lengths.append(0)
                    sampled_to_fridge_sequence_lengths.append(0)
                    sampled_middle_sequence_lengths.append(0)
                    continue

                p_to_idx = np.random.randint(0, len(possible_paths_to))
                p_from_idx = np.random.randint(0, len(possible_paths_from))
                p_to_seq = possible_paths_to[p_to_idx]
                p_from_seq = possible_paths_from[p_from_idx]
                
                full_sequence = []
                if p_to_seq and p_from_seq: # Both paths must be non-empty to combine meaningfully
                    full_sequence = p_to_seq[:-1] + p_from_seq 
                elif p_to_seq: # Only to_fridge path exists
                    full_sequence = p_to_seq
                elif p_from_seq: # Only from_fridge path exists
                    full_sequence = p_from_seq
                # If both are empty, full_sequence remains []
                                
                compressed_audio_for_sample = get_compressed_audio_from_path(self, full_sequence)
                # Basic validation of the compressed audio before appending
                if not (isinstance(compressed_audio_for_sample, list) and len(compressed_audio_for_sample) == 5 and isinstance(compressed_audio_for_sample[0], int) and isinstance(compressed_audio_for_sample[4], int)):
                    logger.warning(f"Agent {agent_id} ({agent_type}) AUDIO: Malformed compressed audio for sampled path. Storing as None. Path: {full_sequence}, Comp: {compressed_audio_for_sample}")
                    sampled_audio_sequences_compressed.append(None)
                else:
                    sampled_audio_sequences_compressed.append(compressed_audio_for_sample)

                sampled_full_sequences.append(full_sequence)
                sampled_to_fridge_sequences.append(p_to_seq)       # P1 (Start -> Fridge)
                sampled_middle_sequences.append(p_from_seq)      # P_FS (Fridge -> Start)
                
                sampled_full_sequence_lengths.append(len(full_sequence) -1 if full_sequence else 0)
                sampled_to_fridge_sequence_lengths.append(len(p_to_seq) -1 if p_to_seq else 0)
                sampled_middle_sequence_lengths.append(len(p_from_seq) -1 if p_from_seq else 0)
        else: # Fallback for unknown evidence type
            logger.error(f"Unknown evidence type '{params.evidence}' in get_sample_paths.")
            # Return empty structure
            return {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': [], 'audio_sequences': [], 'to_fridge_sequences': [], 'full_sequence_lengths': [], 'to_fridge_sequence_lengths': [], 'middle_sequence_lengths': []}

        return {
            'full_sequences': sampled_full_sequences,  
            'middle_sequences': sampled_middle_sequences,   
            'chosen_plant_spots': sampled_chosen_plant_spots, # Empty for audio
            'audio_sequences': sampled_audio_sequences_compressed, # Renamed for clarity from original, now holds compressed
            'to_fridge_sequences': sampled_to_fridge_sequences,
            'full_sequence_lengths': sampled_full_sequence_lengths,
            'to_fridge_sequence_lengths': sampled_to_fridge_sequence_lengths,
            'middle_sequence_lengths': sampled_middle_sequence_lengths 
        }

    def _calculate_optimal_plant_spot_and_slider(self, agent_id, p2_seq, fridge_access_point, naive_A_map, naive_B_map):
        logger = logging.getLogger(__name__)
        optimal_plant_spot = None
        best_slider = -50.0 if agent_id == 'A' else 50.0
        valid_planting_spots = []

        if fridge_access_point and fridge_access_point in p2_seq:
            try:
                fridge_idx_in_p2 = p2_seq.index(fridge_access_point)
                on_return_segment = False
                for i_coord_path, coord_in_path in enumerate(p2_seq): # i_coord_path is index in p2_seq
                    if i_coord_path < fridge_idx_in_p2: continue
                    if i_coord_path == fridge_idx_in_p2: on_return_segment = True
                    if on_return_segment:
                        if coord_in_path in self.node_to_vid: # Check if coord is a valid node
                            vid = self.node_to_vid[coord_in_path]
                            node_attrs = self.igraph.vs[vid]
                            if node_attrs['room'] == 'Kitchen' and not node_attrs['is_door']:
                                valid_planting_spots.append(coord_in_path)
                        else:
                            logger.warning(f"Coordinate {coord_in_path} in p2_seq not in node_to_vid map during plant spot calculation.")
            except ValueError: # fridge_access_point not in p2_seq
                 logger.warning(f"Fridge access point {fridge_access_point} not found in p2_seq: {p2_seq}. Cannot find planting spots.")
                 return None, best_slider
        
        if not valid_planting_spots: return None, best_slider

        current_best_slider_for_agent = best_slider
        chosen_spot_for_this_eval = None
        other_agent_id = 'B' if agent_id == 'A' else 'A'
        other_agent_start_coord = self.start_coords.get(other_agent_id)

        for tile in valid_planting_spots:
            l_A = naive_A_map.get(tuple(tile), 0.0)
            l_B = naive_B_map.get(tuple(tile), 0.0)
            current_slider_at_tile = normalized_slider_prediction(l_A, l_B)

            if agent_id == 'A':
                if current_slider_at_tile > current_best_slider_for_agent:
                    current_best_slider_for_agent = current_slider_at_tile
                    chosen_spot_for_this_eval = tile
                elif current_slider_at_tile == current_best_slider_for_agent:
                    if chosen_spot_for_this_eval is None: chosen_spot_for_this_eval = tile
                    elif other_agent_start_coord:
                        dist_curr = np.linalg.norm(np.array(tile) - np.array(other_agent_start_coord))
                        dist_chosen = np.linalg.norm(np.array(chosen_spot_for_this_eval) - np.array(other_agent_start_coord))
                        if dist_curr < dist_chosen: chosen_spot_for_this_eval = tile
            else: # Agent B
                if current_slider_at_tile < current_best_slider_for_agent:
                    current_best_slider_for_agent = current_slider_at_tile
                    chosen_spot_for_this_eval = tile
                elif current_slider_at_tile == current_best_slider_for_agent:
                    if chosen_spot_for_this_eval is None: chosen_spot_for_this_eval = tile
                    elif other_agent_start_coord:
                        dist_curr = np.linalg.norm(np.array(tile) - np.array(other_agent_start_coord))
                        dist_chosen = np.linalg.norm(np.array(chosen_spot_for_this_eval) - np.array(other_agent_start_coord))
                        if dist_curr < dist_chosen: chosen_spot_for_this_eval = tile
        
        optimal_plant_spot = chosen_spot_for_this_eval
        best_slider = current_best_slider_for_agent
        return optimal_plant_spot, best_slider

    def _get_noisy_plant_spot(self, optimal_spot, sigma):
        potential_spots = [optimal_spot]
        ox, oy = optimal_spot
        cardinal_neighbors = [(ox + 1, oy), (ox - 1, oy), (ox, oy + 1), (ox, oy - 1)]
        for neighbor_coord in cardinal_neighbors:
            # if self.graph.has_node(neighbor): # NX version
            if neighbor_coord in self.node_to_vid: # IG version
                vid = self.node_to_vid[neighbor_coord]
                node_attrs = self.igraph.vs[vid]
                # if node_data.get('room') == 'Kitchen' and not node_data.get('is_door', False): # NX version
                if node_attrs['room'] == 'Kitchen' and not node_attrs['is_door']:
                    potential_spots.append(neighbor_coord)
        
        if len(potential_spots) == 1: return potential_spots[0]

        weights = [np.exp(-((spot[0] - ox)**2 + (spot[1] - oy)**2) / (2 * sigma**2)) for spot in potential_spots]
        probabilities = np.array(weights) / np.sum(weights)
        chosen_idx = np.random.choice(len(potential_spots), p=probabilities)
        return potential_spots[chosen_idx]
    

def load_simple_path_sequences(log_dir_base: str, trial_name: str, w_t0: World, params: SimulationParams, max_steps: int):
    """Loads or computes simple path sequences for a given trial (for agents A and B)."""
    logger = logging.getLogger(__name__)
    # Load from agent A and B tuples from cache if it exists
    if params.evidence == 'audio':
        pickle_path = os.path.join(log_dir_base, 'simple_paths', f'{trial_name}_simple_paths_audio_{max_steps}.pkl')
    else:
        pickle_path = os.path.join(log_dir_base, 'simple_paths', f'{trial_name}_simple_paths_visual_{max_steps}.pkl')

    paths_A = None
    paths_B = None
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            cached_data = pickle.load(f)
            paths_A = cached_data['A']
            paths_B = cached_data['B']

    if paths_A is not None and paths_B is not None:
        logger.info(f"Loaded simple path sequences for {trial_name} from cache.")
        return paths_A, paths_B
                    
    logger.info(f"Computing simple path sequences for {trial_name} (A and B) with max_steps={max_steps}...")
    
    simple_paths_A_p1, simple_paths_A_p2, simple_paths_A_p3, simple_paths_A_fs = w_t0.get_subgoal_simple_path_sequences(
        agent_id='A', params=params, evidence_type=params.evidence, max_steps_middle=max_steps
    )
    simple_paths_B_p1, simple_paths_B_p2, simple_paths_B_p3, simple_paths_B_fs = w_t0.get_subgoal_simple_path_sequences(
        agent_id='B', params=params, evidence_type=params.evidence, max_steps_middle=max_steps
    )
    
    paths_A_tuple = (simple_paths_A_p1, simple_paths_A_p2, simple_paths_A_p3, simple_paths_A_fs)
    paths_B_tuple = (simple_paths_B_p1, simple_paths_B_p2, simple_paths_B_p3, simple_paths_B_fs)

    paths_to_cache = {
        'A': paths_A_tuple,
        'B': paths_B_tuple
    }

    # Save to a single pickle file for both agents
    pickle_path = os.path.join(log_dir_base, 'simple_paths', f'{trial_name}_simple_paths_{params.evidence}_{max_steps}.pkl')
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(paths_to_cache, f)
        logger.info(f"Saved computed simple path sequences to cache for {trial_name}.")
    except Exception as e:
        logger.error(f"Error saving cached paths for {trial_name}: {e}")

    return paths_A_tuple, paths_B_tuple

