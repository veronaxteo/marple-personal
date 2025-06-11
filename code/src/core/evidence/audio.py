"""
Audio evidence processing functionality.

Handles audio token generation, compression, and likelihood calculations
for path-based audio evidence modeling.
"""

import logging
from typing import List
from scipy.stats import norm
from src.cfg import SimulationConfig


# TODO: fix param warnings
def get_audio_tokens_for_path(world_state: 'World', path_coords: List) -> List[str]:
    """
    Generate raw audio tokens for an agent traversing a path.
    
    Args:
        world_state: World object containing environment information
        path_coords: List of (x, y) coordinate tuples representing the path
        
    Returns:
        List of raw audio tokens like ['step', 'step', 'fridge_opened', 'snack_picked_up', 'fridge_closed', 'step']
    """
    raw_tokens = []
    fridge_access_point = world_state.get_fridge_access_point()
    fridge_event_added = False
    
    for i, coord in enumerate(path_coords):
        if i > 0:
            raw_tokens.append('step')

        if coord == fridge_access_point and not fridge_event_added:
            raw_tokens.extend(['fridge_opened', 'snack_picked_up', 'fridge_closed'])
            fridge_event_added = True
    return raw_tokens


def parse_raw_audio_tokens(raw_audio_tokens: List[str]) -> List:
    """
    Compress sequences of raw tokens into [num_steps_to, fridge_events, num_steps_from] format.
    """
    compressed_tokens = []
    current_step_count = 0
    
    for token in raw_audio_tokens:
        if token == 'step':
            current_step_count += 1
        else:
            if current_step_count > 0:
                compressed_tokens.append(current_step_count)
            compressed_tokens.append(token)
            current_step_count = 0
    
    if current_step_count > 0:
        compressed_tokens.append(current_step_count)
    return compressed_tokens
    

def get_compressed_audio_from_path(world_state: 'World', path_coords: List) -> List:
    """
    Convert a world coordinate path to compressed audio token sequence.
    """
    raw_tokens = get_audio_tokens_for_path(world_state, path_coords)
    return parse_raw_audio_tokens(raw_tokens)


def single_segment_audio_likelihood(gt_steps: int, path_steps: int, sigma_factor: float = 0.1) -> float:
    """
    Compute likelihood of observing path_steps given expected ground truth steps (gt_steps). 
    Uses normalized Gaussian PDF.
    
    Args:
        gt_steps: Ground truth number of steps
        path_steps: Observed number of steps in path
        sigma_factor: Standard deviation factor for Gaussian likelihood
        
    Returns:
        Normalized likelihood value between 0 and 1
    """
    sigma = max(1.0, gt_steps * sigma_factor) if gt_steps > 0 else 1.0
    
    # Calculate likelihood and normalize by maximum possible likelihood
    likelihood = norm.pdf(path_steps, loc=gt_steps, scale=sigma)
    max_likelihood = norm.pdf(gt_steps, loc=gt_steps, scale=sigma)
    
    if max_likelihood == 0:
        return 1.0 if likelihood == 0 and path_steps == gt_steps else 0.0

    return likelihood / max_likelihood


def generate_ground_truth_audio_sequences(world: 'World', config: SimulationConfig) -> List[List]:
    """
    Generate ground truth compressed audio sequences for detective predictions.
    Only generates sequences from minimum step counts from closest agent to fridge.
    """
    from src.core.world.graph import get_shortest_path_length
    logger = logging.getLogger(__name__)
    ground_truths = []
    
    # Get world geometry information
    fridge_access_point = world.get_fridge_access_point()
    start_coords = world.start_coords
    node_to_vid = world.world_graph.node_to_vid
    igraph = world.world_graph.igraph

    # Convert coordinates to vertex IDs
    start_A_vid = node_to_vid[start_coords['A']]
    start_B_vid = node_to_vid[start_coords['B']]
    fridge_vid = node_to_vid[fridge_access_point]
    
    # Calculate shortest paths to fridge for both agents
    steps_A_to_fridge = get_shortest_path_length(igraph, start_A_vid, fridge_vid)
    steps_B_to_fridge = get_shortest_path_length(igraph, start_B_vid, fridge_vid)
    steps_A_from_fridge = get_shortest_path_length(igraph, fridge_vid, start_A_vid)
    steps_B_from_fridge = get_shortest_path_length(igraph, fridge_vid, start_B_vid)
    
    # Find minimum feasible steps (closest agent determines starting point)
    min_steps_to_fridge = min(s for s in [steps_A_to_fridge, steps_B_to_fridge] if s is not None)
    min_steps_from_fridge = min(s for s in [steps_A_from_fridge, steps_B_from_fridge] if s is not None)
    
    logger.info(f"Agent shortest paths: A->fridge={steps_A_to_fridge}, B->fridge={steps_B_to_fridge}")
    logger.info(f"Agent shortest paths: fridge->A={steps_A_from_fridge}, fridge->B={steps_B_from_fridge}")
    logger.info(f"Using minimum feasible steps: to_fridge>={min_steps_to_fridge}, from_fridge>={min_steps_from_fridge}")

    # Generate ground truth sequences starting from minimum feasible steps
    step_size = config.evidence.audio_gt_step_size
    max_steps = config.sampling.max_steps
    
    for steps_to in range(min_steps_to_fridge, max_steps + 1, step_size):
        for steps_from in range(min_steps_from_fridge, max_steps + 1, step_size):
            ground_truth_seq = [steps_to, 'fridge_opened', 'snack_picked_up', 'fridge_closed', steps_from]
            ground_truths.append(ground_truth_seq)
    
    logger.info(f"Generated {len(ground_truths)} ground truth audio sequences")
    return ground_truths 
