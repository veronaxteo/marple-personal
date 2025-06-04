"""
Utility functions for path sampling calculations.

Provides helper functions for audio utilities, path grouping, plant spot calculation,
and probability transformations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from src.utils.math_utils import softmax_list_vals, normalized_slider_prediction
from src.core.evidence import single_segment_audio_likelihood
from src.cfg import SimulationConfig, PathSamplingTask


def calculate_audio_utilities(
    task: PathSamplingTask,
    candidate_paths_to_fridge: List,
    candidate_paths_from_fridge: List
) -> Tuple[List[Dict], np.ndarray]:
    """Calculate utilities for audio path length pairs"""
    
    # Group paths by length
    paths_by_len_to, paths_by_len_from = group_paths_by_length(
        candidate_paths_to_fridge, candidate_paths_from_fridge
    )
    
    # Create length pair metadata
    length_pair_metadata = []
    unique_lengths_to = sorted(paths_by_len_to.keys())
    unique_lengths_from = sorted(paths_by_len_from.keys())
    
    for len_to in unique_lengths_to:
        for len_from in unique_lengths_from:
            if paths_by_len_to[len_to] and paths_by_len_from[len_from]:
                length_pair_metadata.append({
                    'eval_steps_to': len_to,
                    'eval_steps_from': len_from,
                    'num_paths_to': len(paths_by_len_to[len_to]),
                    'num_paths_from': len(paths_by_len_from[len_from])
                })
    
    if not length_pair_metadata:
        raise ValueError("No valid length pairs found for audio sampling")
    
    # Calculate costs and utilities
    lengths_combined = np.array([[m['eval_steps_to'], m['eval_steps_from']] for m in length_pair_metadata])
    costs = np.sum(lengths_combined, axis=1)
    rescaled_costs = rescale_costs(costs)
    
    utilities = []
    for i, length_meta in enumerate(length_pair_metadata):
        utility = calculate_single_audio_utility(length_meta, rescaled_costs[i], task)
        utilities.append(utility)
    
    # Convert to probabilities
    probabilities = utilities_to_probabilities(np.array(utilities), task)
    
    return length_pair_metadata, probabilities


def calculate_single_audio_utility(
    length_meta: Dict,
    rescaled_cost: float,
    task: PathSamplingTask
) -> float:
    """Calculate utility for a single audio path length pair"""
    
    if task.agent_type == 'sophisticated':
        eval_steps_to = length_meta['eval_steps_to']
        eval_steps_from = length_meta['eval_steps_from']
        
        # Calculate framing likelihoods using cached computation
        likelihood_A, likelihood_B = compute_audio_framing_likelihoods(
            eval_steps_to, eval_steps_from, task.config
        )
        
        # Calculate framing metric
        framing_metric = normalized_slider_prediction(likelihood_A, likelihood_B)
        framing_metric_scaled = framing_metric / 100.0
        
        # Calculate utility based on agent perspective
        cost_factor = task.config.sampling.cost_weight * (1 - rescaled_cost)
        framing_factor = (1 - task.config.sampling.cost_weight) * framing_metric_scaled
        
        if task.agent_id == 'A':
            utility = cost_factor + framing_factor
        else:  # Agent B
            utility = cost_factor - framing_factor
        
        return utility
    
    elif task.agent_type in ['naive', 'uniform']:
        # Only consider cost for naive/uniform agents
        return task.config.sampling.cost_weight * (1 - rescaled_cost)
    
    else:
        raise ValueError(f"Unknown agent type: {task.agent_type}")


def compute_audio_framing_likelihoods(
    eval_steps_to: int,
    eval_steps_from: int,
    config: SimulationConfig
) -> Tuple[float, float]:
    """
    Compute audio framing likelihoods for a given step pair.
    Uses naive agent model step distributions for sophisticated agent framing calculations.
    """
    logger = logging.getLogger(__name__)
    
    # Use naive agent model distributions if available
    naive_A_to_steps = getattr(config.evidence, 'naive_A_to_fridge_steps_model', [])
    naive_A_from_steps = getattr(config.evidence, 'naive_A_from_fridge_steps_model', [])
    naive_B_to_steps = getattr(config.evidence, 'naive_B_to_fridge_steps_model', [])
    naive_B_from_steps = getattr(config.evidence, 'naive_B_from_fridge_steps_model', [])
    
    if not (naive_A_to_steps and naive_A_from_steps and naive_B_to_steps and naive_B_from_steps):
        logger.warning("Naive agent step models not available for audio framing calculation")
        return 0.5, 0.5  # Default neutral likelihood
    
    sigma_factor = config.evidence.audio_similarity_sigma
    
    # Calculate likelihoods for Agent A
    likelihood_A_to = sum(single_segment_audio_likelihood(eval_steps_to, step, sigma_factor) 
                         for step in naive_A_to_steps) / len(naive_A_to_steps)
    likelihood_A_from = sum(single_segment_audio_likelihood(eval_steps_from, step, sigma_factor) 
                           for step in naive_A_from_steps) / len(naive_A_from_steps)
    likelihood_A = likelihood_A_to * likelihood_A_from
    
    # Calculate likelihoods for Agent B
    likelihood_B_to = sum(single_segment_audio_likelihood(eval_steps_to, step, sigma_factor) 
                         for step in naive_B_to_steps) / len(naive_B_to_steps)
    likelihood_B_from = sum(single_segment_audio_likelihood(eval_steps_from, step, sigma_factor) 
                           for step in naive_B_from_steps) / len(naive_B_from_steps)
    likelihood_B = likelihood_B_to * likelihood_B_from
    
    return likelihood_A, likelihood_B


def group_paths_by_length(
    candidate_paths_to_fridge: List,
    candidate_paths_from_fridge: List
) -> Tuple[Dict[int, List], Dict[int, List]]:
    """Group paths by their length for efficient sampling"""
    
    paths_by_len_to = {}
    for path in candidate_paths_to_fridge:
        length = len(path) - 1 if path else 0  # Convert to step count
        if length not in paths_by_len_to:
            paths_by_len_to[length] = []
        paths_by_len_to[length].append(path)
    
    paths_by_len_from = {}
    for path in candidate_paths_from_fridge:
        length = len(path) - 1 if path else 0  # Convert to step count
        if length not in paths_by_len_from:
            paths_by_len_from[length] = []
        paths_by_len_from[length].append(path)
    
    return paths_by_len_to, paths_by_len_from


def sample_paths_with_lengths(
    paths_by_len_to: Dict[int, List],
    paths_by_len_from: Dict[int, List],
    selected_len_to: int,
    selected_len_from: int
) -> Tuple[Optional[List], Optional[List]]:
    """Sample specific paths with given lengths"""
    
    p_to_seq = None
    p_from_seq = None
    
    if selected_len_to in paths_by_len_to and paths_by_len_to[selected_len_to]:
        p_to_seq = np.random.choice(paths_by_len_to[selected_len_to])
    
    if selected_len_from in paths_by_len_from and paths_by_len_from[selected_len_from]:
        p_from_seq = np.random.choice(paths_by_len_from[selected_len_from])
    
    return p_to_seq, p_from_seq


def utilities_to_probabilities(
    utilities: np.ndarray,
    task: PathSamplingTask
) -> np.ndarray:
    """Convert utilities to sampling probabilities using temperature"""
    
    if task.agent_type == 'sophisticated':
        temperature = task.config.sampling.sophisticated_temp
    elif task.agent_type == 'naive':
        temperature = task.config.sampling.naive_temp
    else:  # TODO: what to do for uniform?
        temperature = task.config.sampling.naive_temp
    
    if temperature <= 0:
        # Deterministic selection of highest utility
        probabilities = np.zeros_like(utilities)
        probabilities[np.argmax(utilities)] = 1.0
        return probabilities
    
    return softmax_list_vals(utilities, temperature)


def rescale_costs(costs: np.ndarray) -> np.ndarray:
    """Rescale costs to [0, 1] range"""
    if len(costs) <= 1:
        return np.zeros_like(costs)
    
    min_cost, max_cost = np.min(costs), np.max(costs)
    if max_cost == min_cost:
        return np.zeros_like(costs)
    
    return (costs - min_cost) / (max_cost - min_cost)


def calculate_optimal_plant_spot_and_slider(
    world,
    agent_id: str,
    p2_seq: List,
    fridge_access_point,
    config: SimulationConfig
):
    """
    Calculate optimal plant spot and framing slider value for a path sequence.
    For sophisticated agents in visual evidence scenarios.
    """
    logger = logging.getLogger(__name__)
    
    if not p2_seq or len(p2_seq) < 2:
        return None, 0
    
    best_slider = 0
    best_coord = None
    
    # Get valid kitchen coordinates from the path (excluding doors and fridge)
    valid_coords = []
    for coord in p2_seq[1:-1]:  # Exclude start and end points
        if coord in world.world_graph.node_to_vid:
            vid = world.world_graph.node_to_vid[coord]
            node_attrs = world.world_graph.igraph.vs[vid]
            is_kitchen = node_attrs['room'] == 'Kitchen'
            is_door = node_attrs['is_door']
            
            if is_kitchen and not is_door and coord != fridge_access_point:
                valid_coords.append(coord)
    
    if not valid_coords:
        logger.debug(f"No valid plant spots found in path sequence for agent {agent_id}")
        return None, 0
    
    # Evaluate each potential plant spot
    for coord in valid_coords:
        try:
            # Use naive agent models if available for framing calculation
            naive_A_visual_map = getattr(config.evidence, 'naive_A_visual_likelihoods_map', {})
            naive_B_visual_map = getattr(config.evidence, 'naive_B_visual_likelihoods_map', {})
            
            if naive_A_visual_map and naive_B_visual_map:
                likelihood_A = naive_A_visual_map.get(coord, 0)
                likelihood_B = naive_B_visual_map.get(coord, 0)
                slider_value = normalized_slider_prediction(likelihood_A, likelihood_B)
            else:
                # Fallback: simple distance-based heuristic
                # (This is a simplified placeholder - actual implementation would use proper likelihood models)
                slider_value = 50  # Neutral position
            
            if (agent_id == 'A' and slider_value > best_slider) or \
               (agent_id == 'B' and slider_value < best_slider):
                best_slider = slider_value
                best_coord = coord
                
        except Exception as e:
            logger.warning(f"Error evaluating plant spot {coord} for agent {agent_id}: {e}")
    
    return best_coord, best_slider


def get_noisy_plant_spot(world, optimal_spot, sigma):
    """
    Add noise to optimal plant spot selection.
    Returns a nearby valid kitchen coordinate.
    """
    logger = logging.getLogger(__name__)
    
    if optimal_spot is None:
        return None
    
    if sigma <= 0:
        return optimal_spot
    
    try:
        # Get neighboring coordinates within sigma radius
        candidates = [optimal_spot]  # Include original as candidate
        
        x_opt, y_opt = optimal_spot
        search_radius = max(1, int(sigma))
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                neighbor_coord = (x_opt + dx, y_opt + dy)
                
                # Check if coordinate is valid and in kitchen
                if neighbor_coord in world.world_graph.node_to_vid:
                    vid = world.world_graph.node_to_vid[neighbor_coord]
                    node_attrs = world.world_graph.igraph.vs[vid]
                    is_kitchen = node_attrs['room'] == 'Kitchen'
                    is_door = node_attrs['is_door']
                    
                    if is_kitchen and not is_door:
                        candidates.append(neighbor_coord)
        
        # Select randomly from candidates
        selected_idx = np.random.choice(len(candidates))
        return candidates[selected_idx]
        
    except Exception as e:
        logger.warning(f"Error adding noise to plant spot {optimal_spot}: {e}")
        return optimal_spot


def is_valid_audio_sequence(audio_seq) -> bool:
    """Validate that audio sequence has correct format"""
    if audio_seq is None:
        return False
    
    if not isinstance(audio_seq, list) or len(audio_seq) != 5:
        return False
    
    # Check format: [steps_to, 'fridge_opened', 'snack_picked_up', 'fridge_closed', steps_from]
    if not isinstance(audio_seq[0], int) or not isinstance(audio_seq[4], int):
        return False
    
    expected_events = ['fridge_opened', 'snack_picked_up', 'fridge_closed']
    if audio_seq[1:4] != expected_events:
        return False
    
    return True 