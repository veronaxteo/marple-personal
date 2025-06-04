import numpy as np
from numpy.random import rand
from scipy.ndimage import gaussian_filter
from collections import deque


def normalized_slider_prediction(value_A, value_B):
    """Calculate normalized slider prediction between two values."""
    if sum([value_A, value_B]) == 0: 
        return 0
    else: 
        return round(100 * (value_B / (value_A + value_B)) - 50, 0)


def softmax_list_vals(vals, temp):
    """Apply softmax with temperature to a list of values."""
    return np.exp(np.array(vals) / temp) / np.sum(np.exp(np.array(vals) / temp), axis=0)


def flip(p):
    """Flip a coin with probability p."""
    if rand() < p:
        return True
    return False


def smooth_likelihood_grid(raw_likelihoods_map: dict, world, sigma: float) -> dict:
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

    possible_crumb_coords = world.get_valid_kitchen_crumb_coords_world()
    if not possible_crumb_coords:
        possible_crumb_coords = raw_likelihoods_map.keys()

    for wc_tuple in possible_crumb_coords:
        kc = world.world_to_kitchen_coords(wc_tuple[0], wc_tuple[1])
        if kc:
            smoothed_likelihoods_map[wc_tuple] = grid_smoothed_likelihoods[kc[1], kc[0]]
            
    return smoothed_likelihoods_map


def smooth_likelihood_grid_connectivity_aware(raw_likelihoods_map: dict, world, sigma_steps: int) -> dict:
    """
    Apply connectivity-aware smoothing that respects the navigation graph structure.
    Spreads likelihood only along valid paths.
    
    Args:
        raw_likelihoods_map: Map from world coordinate tuples to likelihood values
        world: World object containing the navigation graph
        sigma_steps: Number of graph steps over which to spread the smoothing effect
        
    Returns:
        dict: Smoothed likelihood map respecting graph connectivity
    """
    if sigma_steps <= 0:
        return raw_likelihoods_map
    
    # Get valid kitchen coordinates for bounds checking
    valid_coords = set(world.get_valid_kitchen_crumb_coords_world())
    if not valid_coords:
        return raw_likelihoods_map
    
    # Initialize smoothed map with zeros
    smoothed_map = {coord: 0.0 for coord in valid_coords}
    
    # For each coordinate with likelihood, spread it along the graph
    for source_coord, source_likelihood in raw_likelihoods_map.items():
        if source_likelihood == 0 or source_coord not in valid_coords:
            continue
            
        # Perform graph-based diffusion from this source
        diffused_values = _graph_diffusion_from_source(
            source_coord, source_likelihood, world, sigma_steps, valid_coords
        )
        
        # Add diffused values to the smoothed map
        for coord, value in diffused_values.items():
            smoothed_map[coord] += value
    
    return smoothed_map


def _graph_diffusion_from_source(source_coord: tuple, source_likelihood: float, 
                                world, sigma_steps: int, valid_coords: set) -> dict:
    """
    Perform graph-based diffusion from a single source coordinate.
    Uses BFS to spread likelihood values based on graph distance.
    
    Args:
        source_coord: Starting coordinate tuple
        source_likelihood: Initial likelihood value to spread
        world: World object with navigation graph
        sigma_steps: Maximum number of steps to spread
        valid_coords: Set of valid coordinates to consider
        
    Returns:
        dict: Map from coordinates to diffused likelihood values
    """
    # Initialize result with source
    diffused = {source_coord: source_likelihood}
    
    # BFS queue: (coord, distance_from_source)
    queue = deque([(source_coord, 0)])
    visited = {source_coord}
    
    while queue:
        current_coord, distance = queue.popleft()
        
        # Stop if we've reached maximum diffusion distance
        if distance >= sigma_steps:
            continue
            
        # Get neighbors from the navigation graph
        neighbors = _get_graph_neighbors(current_coord, world, valid_coords)
        
        for neighbor_coord in neighbors:
            if neighbor_coord in visited:
                continue
                
            visited.add(neighbor_coord)
            neighbor_distance = distance + 1
            
            # Calculate diffusion weight based on distance
            # Use Gaussian-like decay: stronger effect for closer nodes
            weight = np.exp(-(neighbor_distance ** 2) / (2 * (sigma_steps / 2) ** 2))
            diffused_likelihood = source_likelihood * weight
            
            # Add to diffused map
            if neighbor_coord not in diffused:
                diffused[neighbor_coord] = 0.0
            diffused[neighbor_coord] += diffused_likelihood
            
            # Continue BFS from this neighbor
            queue.append((neighbor_coord, neighbor_distance))
    
    return diffused


def _get_graph_neighbors(coord: tuple, world, valid_coords: set) -> list:
    """
    Get valid neighboring coordinates from the navigation graph.
    
    Args:
        coord: Current coordinate tuple
        world: World object with navigation graph
        valid_coords: Set of valid coordinates to filter by
        
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
