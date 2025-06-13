"""
Various math utilities.
"""

import numpy as np
from numpy.random import rand
from scipy.ndimage import gaussian_filter
from typing import List


def normalized_slider_prediction(value_A: float, value_B: float) -> int:
    """
    Calculate normalized slider prediction between two values, [-50, 50].
    """
    if sum([value_A, value_B]) == 0: 
        return 0
    else: 
        return round(100 * (value_B / (value_A + value_B)) - 50, 0)


def softmax_list_vals(vals: List[float], temp: float) -> List[float]:
    """Apply softmax with temperature to a list of values."""
    return np.exp(np.array(vals) / temp) / np.sum(np.exp(np.array(vals) / temp), axis=0)


def flip(p: float) -> bool:
    """Flip a coin with probability p."""
    if rand() < p:
        return True
    return False


def smooth_likelihoods_old(raw_likelihoods_map: dict, world, sigma: float) -> dict:
    """
    Applies Gaussian smoothing to a grid of likelihoods.

    Returns:
        dict: A dictionary mapping world coordinate tuples to smoothed likelihood values.
    """
    if not sigma > 0:
        return raw_likelihoods_map

    kitchen_width = world.coordinate_mapper.kitchen_width
    kitchen_height = world.coordinate_mapper.kitchen_height
    grid_raw_likelihoods = np.zeros((kitchen_height, kitchen_width), dtype=float)

    for wc_tuple, likelihood in raw_likelihoods_map.items():
        kc = world.world_to_kitchen_coords(wc_tuple[0], wc_tuple[1])
        if kc: 
            grid_raw_likelihoods[kc[1], kc[0]] = likelihood
    
    grid_smoothed_likelihoods = gaussian_filter(grid_raw_likelihoods, sigma=sigma)
    
    smoothed_likelihoods_map = {}

    possible_crumb_coords = world.get_valid_kitchen_crumb_coords()
    if not possible_crumb_coords:
        possible_crumb_coords = raw_likelihoods_map.keys()

    for wc_tuple in possible_crumb_coords:
        kc = world.world_to_kitchen_coords(wc_tuple[0], wc_tuple[1])
        if kc:
            smoothed_likelihoods_map[wc_tuple] = grid_smoothed_likelihoods[kc[1], kc[0]]
            
    return smoothed_likelihoods_map


# TODO: move to other file
def smooth_likelihoods(raw_likelihoods_map: dict, sigma_steps: int, precomputed_neighbors: dict) -> dict:
    """
    Iteratively averages likelihoods with precomputed connected kitchen neighbors.
    """
    current_likelihoods = raw_likelihoods_map.copy()

    coords_to_process = list(raw_likelihoods_map.keys())

    for _ in range(sigma_steps):
        likelihoods_at_step_start = current_likelihoods.copy()
        
        for coord in coords_to_process:
            neighbors = precomputed_neighbors.get(coord, [])
            likelihood_at_coord = likelihoods_at_step_start.get(coord, 0.0)
            neighbor_values = [likelihoods_at_step_start.get(n, 0.0) for n in neighbors]
            
            all_vals_for_avg = [likelihood_at_coord] + neighbor_values
            
            # Update likelihood for current coordinate
            current_likelihoods[coord] = np.mean(all_vals_for_avg)
            
    return current_likelihoods


def compute_all_graph_neighbors(world, valid_coords: list) -> dict:
    """
    Precomputes the connected kitchen neighbors for all valid coordinates.

    Returns:
        dict: A dictionary where keys are coordinate tuples and values are
              lists of their connected kitchen neighbor coordinate tuples.
    """
    precomputed_neighbors = {}
    for coord in valid_coords:
        neighbors = _get_graph_neighbors(coord, world, valid_coords)
        precomputed_neighbors[coord] = neighbors
    return precomputed_neighbors


def _get_graph_neighbors(coord: tuple, world, valid_coords: set) -> list:
    """
    Get valid neighboring coordinates from the navigation graph.
    
    Returns:
        list: List of neighboring coordinate tuples
    """
    neighbors = []
    
    # Check if the coordinate exists in the graph
    if hasattr(world, 'world_graph') and coord in world.world_graph.node_to_vid:
        vid = world.world_graph.node_to_vid[coord]
        neighbor_vids = world.world_graph.igraph.neighbors(vid)
        
        # Create reverse mapping from vid to coordinates if it doesn't exist
        if not hasattr(world.world_graph, 'vid_to_node'):
            world.world_graph.vid_to_node = {v: k for k, v in world.world_graph.node_to_vid.items()}
        
        for neighbor_vid in neighbor_vids:
            neighbor_coord = world.world_graph.vid_to_node.get(neighbor_vid)
            if neighbor_coord is None:
                continue
                
            # Only include kitchen coordinates that are valid for crumbs
            if neighbor_coord in valid_coords:
                if neighbor_coord in world.world_graph.node_to_vid:
                    neighbor_vid_check = world.world_graph.node_to_vid[neighbor_coord]
                    neighbor_attrs = world.world_graph.igraph.vs[neighbor_vid_check]
                    if (neighbor_attrs['room'] == 'Kitchen' and 
                        not neighbor_attrs['is_door']):
                        neighbors.append(neighbor_coord)
    
    return neighbors


# Furniture constants
furniture_size = {
    'bed': [3, 2],
    'sofa': [3, 2],
    'light': [1, 2],
    'table': [3, 2],
    'side_table': [1, 1],
    'electric_refrigerator': [2, 3],
    'tv': [2, 2]
} 
