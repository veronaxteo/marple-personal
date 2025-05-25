import json
import networkx as nx
import numpy as np
import time
import logging
import os
from joblib import Parallel, delayed
from utils import softmax_list_vals, normalized_slider_prediction, get_shortest_paths, get_simple_paths
from globals import furniture_size


class World():
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
        self.graph = nx.Graph()
        self.kitchen_info = None
        self.kitchen_width = 0
        self.kitchen_height = 0
        self.kitchen_top_x = 0
        self.kitchen_top_y = 0
        self.start_coords = {}
        self.mission = None
        self._valid_kitchen_crumb_coords_world = None 
        self.create_world()


    def create_world(self):
        """Builds the NetworkX graph and stores key info."""
        self.width = self.info['width']
        self.height = self.info['height']
        self.mission = self.info['agents']['initial'][0]['cur_mission']

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

        # Create graph
        for r in self.info['rooms']['initial']:
            room_graph = nx.grid_2d_graph(
                range(r['top'][0], r['top'][0] + r['size'][0]),
                range(r['top'][1], r['top'][1] + r['size'][1]),
                create_using=nx.Graph
            )
            nodes_to_remove = {loc for loc in room_graph.nodes if self.is_furniture_at(loc)}
            room_graph.remove_nodes_from(nodes_to_remove)
            for node in room_graph.nodes:
                 room_graph.nodes[node]['is_door'] = False
                 room_graph.nodes[node]['room'] = r['type']
            self.graph = nx.compose(self.graph, room_graph)

        for d in self.info['doors']['initial']:
            door_pos = tuple(d['pos'])
            self.graph.add_node(door_pos, is_door=True, state=d['state'], room=None)
            potential_neighbors = []
            if d['dir'] == 'horz':
                potential_neighbors = [(door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1)]
            elif d['dir'] == 'vert':
                potential_neighbors = [(door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1])]
            for neighbor_pos in potential_neighbors:
                if self.graph.has_node(neighbor_pos):
                    self.graph.add_edge(door_pos, neighbor_pos)


    @staticmethod
    def initialize_world_start(filename):
        """Initializes World from trial JSON."""
        json_path = f'../trials/suspect/json/{filename}'
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Trial JSON file not found: {json_path}")
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
        if not self.graph.has_node(fridge_access_point): raise ValueError(f"Calculated Fridge access point {fridge_access_point} is not a valid node in the graph.")
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
                if self.graph.has_node(coord_tuple) and not self.graph.nodes[coord_tuple].get('is_door', False):
                    valid_coords.append(coord_tuple)
        self._valid_kitchen_crumb_coords_world = sorted(valid_coords)
        return self._valid_kitchen_crumb_coords_world


    def get_closest_door_to_agent(self, agent_id):
        """Finds closest door node to agent start (world coords)."""
        agent_start_pos = self.start_coords.get(agent_id)
        if not agent_start_pos: return None
        door_nodes = [n for n, data in self.graph.nodes(data=True) if data.get('is_door')]
        if not door_nodes: return None
        closest_door = None
        min_dist = float('inf')
        for door_pos in door_nodes:
            try:
                dist = nx.shortest_path_length(self.graph, source=agent_start_pos, target=door_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_door = door_pos
            except nx.NetworkXNoPath: continue
        return closest_door


    def get_subgoals(self, agent_id):
        """Get subgoals sequence (world coords tuples) for an agent."""
        start_pos = self.start_coords.get(agent_id)
        if not start_pos: raise ValueError(f"Start pos not found for agent {agent_id}")
        subgoals = [start_pos]
        if self.mission == 'get_snack':
            fridge_info = next((f for f in self.kitchen_info['furnitures']['initial'] if f['type'] == 'electric_refrigerator'), None)
            if fridge_info is None: raise ValueError("Refrigerator not found.")
            fp = fridge_info['pos']
            fridge_access_point = (fp[0] - 1, fp[1] + 2)
            if not self.graph.has_node(fridge_access_point): raise ValueError(f"Fridge access point {fridge_access_point} invalid.")
            door_node = self.get_closest_door_to_agent(agent_id)
            if door_node is None: raise ValueError(f"No reachable door found for agent {agent_id}.")
            subgoals.extend([fridge_access_point, door_node])
        subgoals.append(start_pos)
        return subgoals


    def get_subgoal_simple_path_sequences(self, agent_id, max_steps_middle):
        """Get simple path sequences (lists of world coord tuples) for subgoals."""
        start_time = time.time(); logger = logging.getLogger(__name__)
        try:
            subgoals = self.get_subgoals(agent_id)
            logger.info(f"Subgoals for agent {agent_id}: {subgoals}")
            if len(subgoals) < 4: raise ValueError("Insufficient subgoals.")
        except ValueError as e: logger.error(f"Error getting subgoals for {agent_id}: {e}"); return [[], [], []]
        results = Parallel(n_jobs=-1)(
            [delayed(get_shortest_paths)(self.graph, subgoals[0], subgoals[1]),
             delayed(get_simple_paths)(self.graph, subgoals[1], subgoals[2], max_steps_middle),
             delayed(get_shortest_paths)(self.graph, subgoals[2], subgoals[3])]
        )
        sequences_p1 = sorted(results[0]) if results[0] else []
        sequences_p2 = sorted(results[1]) if results[1] else []
        sequences_p3 = sorted(results[2]) if results[2] else []
        logger.info(f"Paths per segment {agent_id}: {len(sequences_p1)}, {len(sequences_p2)}, {len(sequences_p3)}")
        logger.info(f"Pathfinding time ({agent_id}): {time.time() - start_time:.2f}s")
        return [sequences_p1, sequences_p2, sequences_p3]


    def get_sample_paths(self, agent_id: str, simple_path_sequences: list, num_sample_paths: int, agent_type: str, 
                        naive_A_crumb_likelihoods_map: dict, naive_B_crumb_likelihoods_map: dict, 
                        w: float, temp: float, noisy_planting_sigma: float):
        """
        Samples full agent paths, returning sequences, middle sequences, and numbered 2D arrays.
        For sophisticated agents, it now also determines an optimal crumb planting spot for each path
        and returns a list of these spots.
        If noisy_planting_sigma is greater than 0 for sophisticated agents, the final chosen spot might be a Gaussian-weighted neighbor of the optimal.
        """
        start_time = time.time()
        logger = logging.getLogger(__name__)

        sequences_p1, sequences_p2, sequences_p3 = simple_path_sequences

        sampled_full_sequences = []
        sampled_middle_sequences = []
        sampled_numbered_arrays = []
        sampled_chosen_plant_spots = []

        # Compute likelihoods
        likelihoods = None
        if agent_type != 'uniform':
            middle_path_lengths = np.array([len(seq) for seq in sequences_p2])
            min_len, max_len = np.min(middle_path_lengths), np.max(middle_path_lengths)
            rescaled_lengths = np.zeros_like(middle_path_lengths, dtype=float)  # note these are len(middle_path)
            if max_len > min_len:
                rescaled_lengths = (middle_path_lengths - min_len) / (max_len - min_len)  # rescale to [0, 1]

            # Crumb planting
            path_utilities = []
            path_optimal_plant_spots = [] 

            if agent_type == 'sophisticated':
                fridge_access_point = self.get_fridge_access_point()

                for idx, p2_seq in enumerate(sequences_p2):
                    best_achievable_slider_for_path = -50.0 if agent_id == 'A' else 50.0
                    optimal_plant_spot_for_this_p2_seq = None
                    on_return_segment = False
                    valid_planting_spots_on_path = []

                    if fridge_access_point:
                        fridge_idx_in_p2 = p2_seq.index(fridge_access_point)
                
                        # Valid planting spots are kitchen tiles on the path the fridge_access_point
                        for i_coord, coord in enumerate(p2_seq):
                            if i_coord < fridge_idx_in_p2: 
                                continue
                            
                            # Allow planting at fridge access point
                            if i_coord == fridge_idx_in_p2: # at fridge
                                on_return_segment = True
                            
                            if on_return_segment:
                                node_data = self.graph.nodes.get(coord, {})
                                if node_data.get('room') == 'Kitchen' and not node_data.get('is_door', False):
                                    valid_planting_spots_on_path.append(coord)
                    
                        current_best_slider_for_agent = -50.0 if agent_id == 'A' else 50.0 
                        chosen_spot_for_this_eval = None
                        
                        # Determine other agent's start coordinate for tie-breaking
                        other_agent_id = 'B' if agent_id == 'A' else 'A'
                        other_agent_start_coord = self.start_coords.get(other_agent_id)
                        if other_agent_start_coord is None:
                            logger.warning(f"Could not find start coordinates for other agent {other_agent_id} for tie-breaking. Defaulting to old tie-breaking.")

                        # TODO: clean and make concise
                        for tile in valid_planting_spots_on_path:
                            l_A_at_tile = naive_A_crumb_likelihoods_map.get(tile, 0.0)
                            l_B_at_tile = naive_B_crumb_likelihoods_map.get(tile, 0.0)
                            
                            current_slider_at_tile = normalized_slider_prediction(l_A_at_tile, l_B_at_tile)

                            if agent_id == 'A':
                                if current_slider_at_tile > current_best_slider_for_agent:
                                    current_best_slider_for_agent = current_slider_at_tile
                                    chosen_spot_for_this_eval = tile

                                elif current_slider_at_tile == current_best_slider_for_agent:
                                    if chosen_spot_for_this_eval is None: 
                                        chosen_spot_for_this_eval = tile

                                    elif other_agent_start_coord: # Tie-break using distance to other agent
                                        dist_current_tile_to_other = np.linalg.norm(np.array(tile) - np.array(other_agent_start_coord))
                                        dist_chosen_spot_to_other = np.linalg.norm(np.array(chosen_spot_for_this_eval) - np.array(other_agent_start_coord))
                                        if dist_current_tile_to_other < dist_chosen_spot_to_other:
                                            chosen_spot_for_this_eval = tile
                                    
                            elif agent_id == 'B': 
                                if current_slider_at_tile < current_best_slider_for_agent:
                                    current_best_slider_for_agent = current_slider_at_tile
                                    chosen_spot_for_this_eval = tile
                                elif current_slider_at_tile == current_best_slider_for_agent:
                                    if chosen_spot_for_this_eval is None:
                                        chosen_spot_for_this_eval = tile
                                    elif other_agent_start_coord:
                                        dist_current_tile_to_other = np.linalg.norm(np.array(tile) - np.array(other_agent_start_coord))
                                        dist_chosen_spot_to_other = np.linalg.norm(np.array(chosen_spot_for_this_eval) - np.array(other_agent_start_coord))
                                        if dist_current_tile_to_other < dist_chosen_spot_to_other:
                                            chosen_spot_for_this_eval = tile
                                    
                        
                        best_achievable_slider_for_path = current_best_slider_for_agent
                        # print(f"best_achievable_slider_for_path: {best_achievable_slider_for_path}")
                        optimal_plant_spot_for_this_p2_seq = chosen_spot_for_this_eval
                        # print(f"optimal_plant_spot_for_this_p2_seq: {optimal_plant_spot_for_this_p2_seq}")
                        # breakpoint()
                    
                    path_optimal_plant_spots.append(optimal_plant_spot_for_this_p2_seq)
                    path_framing_metric_scaled = best_achievable_slider_for_path / 100.0  # rescale to [-0.5, 0.5]
                    
                    current_rescaled_length = rescaled_lengths[idx]

                    if agent_id == 'A':
                        utility = w * (1 - current_rescaled_length) + (1 - w) * path_framing_metric_scaled
                        cost = w * current_rescaled_length + (1 - w) * (1 - path_framing_metric_scaled)
                    else: # Agent B
                        utility = w * (1 - current_rescaled_length) - (1 - w) * path_framing_metric_scaled
                        cost = w * current_rescaled_length + (1 - w) * path_framing_metric_scaled
                    path_utilities.append(utility)

            elif agent_type == 'naive':
                for l_rescaled in rescaled_lengths:
                    utility = w * (1 - l_rescaled)
                    path_utilities.append(utility)
                path_optimal_plant_spots = [None] * len(sequences_p2)
            
            else:  # no planting
                path_utilities = [1.0] * len(sequences_p2)
                path_optimal_plant_spots = [None] * len(sequences_p2)

            likelihoods = softmax_list_vals(np.array(path_utilities), temp=temp)

        # Sample paths
        num_first = len(sequences_p1)
        num_middle = len(sequences_p2)
        num_last = len(sequences_p3)

        for _ in range(num_sample_paths):
            # Sample indices
            idx1 = np.random.randint(0, num_first)
            idx3 = np.random.randint(0, num_last)
            if agent_type == 'uniform':
                idx2 = np.random.randint(0, num_middle)  # uniform sampling
            else:
                idx2 = np.random.choice(num_middle, p=likelihoods)  # weighted sampling according to likelihoods

            # Retrieve paths
            p1_seq = sequences_p1[idx1]
            p2_seq = sequences_p2[idx2]
            p3_seq = sequences_p3[idx3]

            sampled_middle_sequences.append(p2_seq)
            
            # Determine the chosen plant spot for this sampled path
            optimal_spot_for_chosen_path = None
            # path_optimal_plant_spots contains the best spot for each p2_seq according to agent's reasoning
            if agent_type == 'sophisticated' or agent_type == 'naive': 
                if idx2 < len(path_optimal_plant_spots):
                    optimal_spot_for_chosen_path = path_optimal_plant_spots[idx2]
               
            # Apply noisy planting if enabled for sophisticated agent
            if agent_type == 'sophisticated' and optimal_spot_for_chosen_path is not None and noisy_planting_sigma > 0:
                final_chosen_spot_for_path = optimal_spot_for_chosen_path 
                potential_noisy_spots = [optimal_spot_for_chosen_path]
                ox, oy = optimal_spot_for_chosen_path   
                
                cardinal_neighbors_coords = [
                    (ox + 1, oy), (ox - 1, oy),
                    (ox, oy + 1), (ox, oy - 1)
                ]
                
                for neighbor_coord in cardinal_neighbors_coords:
                    if self.graph.has_node(neighbor_coord):
                        node_data = self.graph.nodes[neighbor_coord]
                        if node_data.get('room') == 'Kitchen' and not node_data.get('is_door', False):
                            potential_noisy_spots.append(neighbor_coord)

                if len(potential_noisy_spots) == 1:
                    final_chosen_spot_for_path = potential_noisy_spots[0]

                elif potential_noisy_spots: 
                    weights = []
                    sigma_execution_noise = noisy_planting_sigma
                    
                    for spot in potential_noisy_spots:
                        dist_sq = (spot[0] - ox)**2 + (spot[1] - oy)**2 
                        weight = np.exp(-dist_sq / (2 * sigma_execution_noise**2))
                        weights.append(weight)
                    
                    # Normalize weights to get probabilities
                    probabilities = np.array(weights) / np.sum(weights)
                    chosen_idx = np.random.choice(len(potential_noisy_spots), p=probabilities)
                    final_chosen_spot_for_path = potential_noisy_spots[chosen_idx]

                sampled_chosen_plant_spots.append(final_chosen_spot_for_path)
            elif agent_type == 'sophisticated': 
                sampled_chosen_plant_spots.append(optimal_spot_for_chosen_path)
            else: 
                sampled_chosen_plant_spots.append(None) 

            # Construct full path
            full_sequence = p1_seq[:-1] + p2_seq[:-1] + p3_seq  # remove last element from each sequence (duplicates)
            sampled_full_sequences.append(full_sequence)

            # Generate numbered 2D array (kitchen coords)
            numbered_grid = np.zeros((self.kitchen_height, self.kitchen_width), dtype=np.int16)
            for step, world_coord in enumerate(full_sequence, 1):
                kitchen_coord = self.world_to_kitchen_coords(world_coord[0], world_coord[1])
                if kitchen_coord is not None:
                    kx, ky = kitchen_coord
                    numbered_grid[ky, kx] = step 
            sampled_numbered_arrays.append(numbered_grid)

        return {
            'full_sequences': sampled_full_sequences,  
            'middle_sequences': sampled_middle_sequences,   
            'numbered_arrays': sampled_numbered_arrays,    
            'chosen_plant_spots': sampled_chosen_plant_spots
        }
