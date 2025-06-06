"""
Utility functions for path sampling calculations.

Provides helper functions for audio utilities, path grouping, plant spot calculation,
and probability transformations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set

from src.utils.math_utils import softmax_list_vals, normalized_slider_prediction
from src.core.evidence.audio import single_segment_audio_likelihood
from src.cfg import SimulationConfig, PathSamplingTask


def calculate_visual_utilities(task: PathSamplingTask, sequences_p2: List) -> Tuple[np.ndarray, List]:
    """Calculate utilities for visual path segments (P2: Fridge->Door)."""
    middle_path_lengths = np.array([len(seq) for seq in sequences_p2])
    
    min_len = np.min(middle_path_lengths) if len(middle_path_lengths) > 0 else 0
    max_len = np.max(middle_path_lengths) if len(middle_path_lengths) > 0 else 0
    rescaled_lengths = np.zeros_like(middle_path_lengths, dtype=float)
    if max_len > min_len:
        rescaled_lengths = (middle_path_lengths - min_len) / (max_len - min_len)
    
    optimal_plant_spots = [None] * len(sequences_p2)
    utilities = []
    
    if task.agent_type == 'sophisticated':
        fridge_access_point = task.world.get_fridge_access_point()
        for idx, p2_seq in enumerate(sequences_p2):
            optimal_spot, best_slider = calculate_optimal_plant_spot_and_slider(
                task.world, task.agent_id, p2_seq, fridge_access_point, task.config
            )
            optimal_plant_spots[idx] = optimal_spot
            
            path_framing_metric_scaled = best_slider / 100.0
            utility_factor = task.config.sampling.cost_weight * (1 - rescaled_lengths[idx])
            
            if task.agent_id == 'A':
                utility = utility_factor + (1 - task.config.sampling.cost_weight) * path_framing_metric_scaled
            else:
                utility = utility_factor - (1 - task.config.sampling.cost_weight) * path_framing_metric_scaled
            utilities.append(utility)
    
    elif task.agent_type in ['naive', 'uniform']:
        # Naive agents just prefer shorter paths (higher utility for lower cost)
        utilities = [1.0 - l for l in rescaled_lengths]
    
    return np.array(utilities), optimal_plant_spots


def group_paths_by_length(paths: List) -> Dict[int, List]:
    """Groups a list of paths by their length."""
    paths_by_len = {}
    for path in paths:
        length = len(path) - 1 if path else 0
        if length not in paths_by_len:
            paths_by_len[length] = []
        paths_by_len[length].append(path)
    return paths_by_len


def calculate_audio_utilities(
    task: PathSamplingTask,
    candidate_paths_to_fridge: List,
    candidate_paths_from_fridge: List
) -> Tuple[List[Dict], np.ndarray]:
    """Calculate utilities for audio path length pairs efficiently."""
    
    paths_by_len_to = group_paths_by_length(candidate_paths_to_fridge)
    paths_by_len_from = group_paths_by_length(candidate_paths_from_fridge)
    
    unique_lengths_to = sorted(paths_by_len_to.keys())
    unique_lengths_from = sorted(paths_by_len_from.keys())
    
    length_pair_metadata = []
    # Create metadata for each valid pair of lengths
    for len_to in unique_lengths_to:
        for len_from in unique_lengths_from:
            length_pair_metadata.append({
                'eval_steps_to': len_to,
                'eval_steps_from': len_from,
            })
    
    if not length_pair_metadata:
        return [], np.array([])

    # Calculate costs and utilities for each length pair
    lengths_combined = np.array([[m['eval_steps_to'], m['eval_steps_from']] for m in length_pair_metadata])
    costs = np.sum(lengths_combined, axis=1)
    rescaled_costs = rescale_costs(costs)
    
    utilities = [
        calculate_single_audio_utility(length_meta, rescaled_costs[i], task)
        for i, length_meta in enumerate(length_pair_metadata)
    ]
    
    probabilities = utilities_to_probabilities(np.array(utilities), task)
    return length_pair_metadata, probabilities


def calculate_single_audio_utility(
    length_meta: Dict,
    rescaled_cost: float,
    task: PathSamplingTask
) -> float:
    """Calculate utility for a single audio path length pair."""
    
    if task.agent_type == 'sophisticated':
        eval_steps_to = length_meta['eval_steps_to']
        eval_steps_from = length_meta['eval_steps_from']
        
        likelihood_A, likelihood_B = compute_audio_framing_likelihoods(eval_steps_to, eval_steps_from, task.config)
        
        framing_metric = normalized_slider_prediction(likelihood_A, likelihood_B)
        framing_metric_scaled = framing_metric / 100.0 
        
        cost_factor = task.config.sampling.cost_weight * (1 - rescaled_cost)
        framing_factor = (1 - task.config.sampling.cost_weight) * framing_metric_scaled
        
        if task.agent_id == 'A':
            utility = cost_factor + framing_factor
        else:
            utility = cost_factor - framing_factor
        
        return utility
    
    elif task.agent_type in ['naive', 'uniform']:
        return task.config.sampling.cost_weight * (1 - rescaled_cost)


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
    
    naive_A_to_steps = getattr(config.evidence, 'naive_A_to_fridge_steps_model', [])
    naive_A_from_steps = getattr(config.evidence, 'naive_A_from_fridge_steps_model', [])
    naive_B_to_steps = getattr(config.evidence, 'naive_B_to_fridge_steps_model', [])
    naive_B_from_steps = getattr(config.evidence, 'naive_B_from_fridge_steps_model', [])
    
    sigma_factor = config.evidence.audio_similarity_sigma
    
    likelihood_A_to = sum(single_segment_audio_likelihood(eval_steps_to, step, sigma_factor) 
                         for step in naive_A_to_steps) / len(naive_A_to_steps) if naive_A_to_steps else 0
    likelihood_A_from = sum(single_segment_audio_likelihood(eval_steps_from, step, sigma_factor) 
                           for step in naive_A_from_steps) / len(naive_A_from_steps) if naive_A_from_steps else 0
    likelihood_A = likelihood_A_to * likelihood_A_from
    
    likelihood_B_to = sum(single_segment_audio_likelihood(eval_steps_to, step, sigma_factor) 
                         for step in naive_B_to_steps) / len(naive_B_to_steps) if naive_B_to_steps else 0
    likelihood_B_from = sum(single_segment_audio_likelihood(eval_steps_from, step, sigma_factor) 
                           for step in naive_B_from_steps) / len(naive_B_from_steps) if naive_B_from_steps else 0
    likelihood_B = likelihood_B_to * likelihood_B_from
    
    return likelihood_A, likelihood_B


def utilities_to_probabilities(
    utilities: np.ndarray,
    task: PathSamplingTask
) -> np.ndarray:
    """Convert utilities to sampling probabilities using softmax function."""
    
    if task.agent_type == 'sophisticated':
        temperature = task.config.sampling.sophisticated_temp
    elif task.agent_type == 'naive':
        temperature = task.config.sampling.naive_temp
    else:
        temperature = task.config.sampling.naive_temp
    
    if len(utilities) == 0:
        return np.array([])

    if temperature <= 0:
        probabilities = np.zeros_like(utilities, dtype=float)
        if len(probabilities) > 0:
            probabilities[np.argmax(utilities)] = 1.0
        return probabilities
    
    return softmax_list_vals(utilities, temperature)


def rescale_costs(costs: np.ndarray) -> np.ndarray:
    """Rescale costs to [0, 1] range."""
    if len(costs) <= 1:
        return np.zeros_like(costs, dtype=float)
    
    min_cost, max_cost = np.min(costs), np.max(costs)
    if max_cost == min_cost:
        return np.zeros_like(costs, dtype=float)
    
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
    
    valid_coords = []
    for coord in p2_seq[1:-1]:
        if coord in world.world_graph.node_to_vid:
            vid = world.world_graph.node_to_vid[coord]
            node_attrs = world.world_graph.igraph.vs[vid]
            is_kitchen = node_attrs['room'] == 'Kitchen'
            is_door = node_attrs['is_door']
            
            if is_kitchen and not is_door and coord != fridge_access_point:
                valid_coords.append(coord)
    
    if not valid_coords:
        return None, 0.0

    naive_A_visual_map = getattr(config.evidence, 'naive_A_visual_likelihoods_map', {})
    naive_B_visual_map = getattr(config.evidence, 'naive_B_visual_likelihoods_map', {})

    if not naive_A_visual_map or not naive_B_visual_map:
        logger.warning(f"Naive agent visual likelihoods not available for sophisticated agent. Cannot choose optimal plant spot.")
        return None, 0.0

    slider_values = {
        coord: normalized_slider_prediction(
            naive_A_visual_map.get(coord, 0),
            naive_B_visual_map.get(coord, 0)
        ) for coord in valid_coords
    }

    if not slider_values:
        return None, 0.0

    if agent_id == 'A':
        best_coord = max(slider_values, key=slider_values.get)
    else:
        best_coord = min(slider_values, key=slider_values.get)
        
    best_slider = slider_values[best_coord]
    
    return best_coord, best_slider


def get_noisy_plant_spot(optimal_spot: Tuple[int, int], 
                         sigma: float, 
                         valid_plant_spots: Set[Tuple[int, int]]) -> Tuple[int, int]:
    logger = logging.getLogger(__name__)

    if sigma <= 0:
        return optimal_spot

    candidate_spots = []
    
    search_radius_grid_units = max(0, int(sigma))

    min_x = optimal_spot[0] - search_radius_grid_units
    max_x = optimal_spot[0] + search_radius_grid_units
    min_y = optimal_spot[1] - search_radius_grid_units
    max_y = optimal_spot[1] + search_radius_grid_units

    for r_coord in range(min_x, max_x + 1):
        for c_coord in range(min_y, max_y + 1):
            potential_spot = (r_coord, c_coord)
            if potential_spot in valid_plant_spots:
                candidate_spots.append(potential_spot)
    
    if not candidate_spots:
        return optimal_spot

    weights = []
    for spot in candidate_spots:
        distance_sq = (spot[0] - optimal_spot[0])**2 + (spot[1] - optimal_spot[1])**2
        weight = np.exp(-distance_sq / (2 * (sigma**2 + 1e-6))) 
        weights.append(weight)

    sum_weights = np.sum(weights)
    if sum_weights == 0:
        logger.warning(f"All Gaussian weights are zero for {optimal_spot} with sigma {sigma}. Returning optimal_spot.")
        if optimal_spot in candidate_spots:
            return optimal_spot
        return candidate_spots[0]

    probabilities = np.array(weights) / sum_weights
    
    selected_idx = np.random.choice(len(candidate_spots), p=probabilities)
    return candidate_spots[selected_idx]


def is_valid_audio_sequence(audio_seq) -> bool:
    """Validate that audio sequence has correct format"""
    if audio_seq is None:
        return False
    
    if not isinstance(audio_seq, list) or len(audio_seq) != 5:
        return False
    
    if not isinstance(audio_seq[0], int) or not isinstance(audio_seq[4], int):
        return False
    
    expected_events = ['fridge_opened', 'snack_picked_up', 'fridge_closed']
    if audio_seq[1:4] != expected_events:
        return False
    
    return True
