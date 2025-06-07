"""
Utility functions for path sampling calculations.

Provides helper functions for audio utilities, path grouping, plant spot calculation,
and probability transformations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Set
from multiprocessing import Pool

from src.utils.math_utils import softmax_list_vals, normalized_slider_prediction
from src.core.evidence.audio import single_segment_audio_likelihood
from src.cfg import SimulationConfig, PathSamplingTask


def calculate_visual_utilities(task: PathSamplingTask, sequences_p2: List) -> Tuple[np.ndarray, List]:
    """Calculate utilities for visual path segments (Fridge --> Door)."""
    middle_path_lengths = np.array([len(seq) for seq in sequences_p2])
    
    min_len = np.min(middle_path_lengths) if len(middle_path_lengths) > 0 else 0
    max_len = np.max(middle_path_lengths) if len(middle_path_lengths) > 0 else 0
    rescaled_lengths = np.zeros_like(middle_path_lengths, dtype=float)
    if max_len > min_len:
        rescaled_lengths = (middle_path_lengths - min_len) / (max_len - min_len)
    
    optimal_plant_spots = [None] * len(sequences_p2)
    utilities = []
    
    if task.agent_type == 'sophisticated':
        visual_slider_map = getattr(task.config.evidence, 'visual_slider_map', {})
        if not visual_slider_map:
            logging.warning("Visual slider map not found for sophisticated agent.")

        for idx, p2_seq in enumerate(sequences_p2):
            optimal_spot, best_slider = _find_best_framing_spot(p2_seq, task.agent_id, visual_slider_map)
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


def calculate_multimodal_utilities(
    task: PathSamplingTask,
    paths_to_fridge: List,
    paths_from_fridge: List
) -> Tuple[np.ndarray, List, List]:
    """Calculate utilities for multimodal path pairs for sophisticated agents."""
    logger = logging.getLogger(__name__)
    path_pairs = [(p_to, p_from) for p_to in paths_to_fridge for p_from in paths_from_fridge]
    
    if not path_pairs:
        return np.array([]), [], []

    # Get pre-computed slider maps
    visual_slider_map = getattr(task.config.evidence, 'visual_slider_map', {})
    audio_to_slider_map = getattr(task.config.evidence, 'audio_to_slider_map', {})
    audio_from_slider_map = getattr(task.config.evidence, 'audio_from_slider_map', {})

    # Calculate costs and framing metrics for all path pairs
    all_costs = []
    all_framing_metrics = []
    all_optimal_spots = []

    for p_to, p_from in path_pairs:
        # Cost factor
        cost = (len(p_to) - 1) + (len(p_from) - 1)
        all_costs.append(cost)

        # Audio framing
        len_to = len(p_to) - 1
        len_from = len(p_from) - 1
        slider_to = audio_to_slider_map.get(len_to, 0)
        slider_from = audio_from_slider_map.get(len_from, 0)
        framing_metric_audio = (slider_to + slider_from) / 2.0

        # Visual framing
        optimal_spot, best_visual_slider = _find_best_framing_spot(p_from, task.agent_id, visual_slider_map)
        all_optimal_spots.append(optimal_spot)
        framing_metric_visual = best_visual_slider

        # Combine framing metrics
        framing_metric = (framing_metric_audio + framing_metric_visual) / 2.0
        all_framing_metrics.append(framing_metric / 100.0)

    # Rescale all costs to [0, 1]
    rescaled_costs = rescale_costs(np.array(all_costs))

    # Calculate final utilities using rescaled costs
    utilities = []
    for i in range(len(path_pairs)):
        cost_factor = task.config.sampling.cost_weight * (1 - rescaled_costs[i])
        framing_factor = (1 - task.config.sampling.cost_weight) * all_framing_metrics[i]
        
        if task.agent_id == 'A':
            utility = cost_factor + framing_factor
        else:
            utility = cost_factor - framing_factor
        utilities.append(utility)
        
    return np.array(utilities), all_optimal_spots, path_pairs


def calculate_to_fridge_utilities(task: PathSamplingTask, paths_to_fridge: List) -> np.ndarray:
    """Calculate utilities for 'to fridge' paths using pre-computed slider map."""
    audio_to_slider_map = getattr(task.config.evidence, 'audio_to_slider_map', {})

    costs = np.array([len(p) - 1 for p in paths_to_fridge])
    rescaled_costs = rescale_costs(costs)
    
    utilities = []
    for i, path in enumerate(paths_to_fridge):
        path_len = len(path) - 1
        framing_metric = audio_to_slider_map.get(path_len, 0) / 100.0
        
        cost_factor = task.config.sampling.cost_weight * (1 - rescaled_costs[i])
        framing_factor = (1 - task.config.sampling.cost_weight) * framing_metric

        if task.agent_id == 'A':
            utility = cost_factor + framing_factor
        else:
            utility = cost_factor - framing_factor
        utilities.append(utility)

    return np.array(utilities)


def _calculate_utility_for_single_from_path(args: Tuple) -> Tuple[float, any]:
    """
    Worker function to calculate utility for a single 'from' path for multimodal agents.
    """
    i, p_from, task, rescaled_costs = args
    
    config = task.config
    agent_id = task.agent_id
    
    # Get pre-computed slider maps
    visual_slider_map = getattr(config.evidence, 'visual_slider_map', {})
    audio_from_slider_map = getattr(config.evidence, 'audio_from_slider_map', {})

    # Visual evidence framing
    optimal_spot, best_visual_slider = _find_best_framing_spot(p_from, agent_id, visual_slider_map)
    framing_metric_visual = best_visual_slider / 100.0
    
    # Audio evidence framing
    len_from = len(p_from) - 1
    framing_metric_audio = audio_from_slider_map.get(len_from, 0) / 100.0
    
    # Combine framing scores by averaging them
    # TODO: is this what we want? Or multiply them?
    framing_metric = (framing_metric_visual + framing_metric_audio) / 2
    cost_factor = config.sampling.cost_weight * (1 - rescaled_costs[i])
    framing_factor = (1 - config.sampling.cost_weight) * framing_metric

    if agent_id == 'A':
        utility = cost_factor + framing_factor
    else:
        utility = cost_factor - framing_factor
        
    return utility, optimal_spot


def calculate_from_fridge_utilities(task: PathSamplingTask, paths_from_fridge: List) -> Tuple[np.ndarray, List]:
    """
    Calculate utilities for 'from fridge' paths using parallel processing.
    """
    costs = np.array([len(p) - 1 for p in paths_from_fridge])
    rescaled_costs = rescale_costs(costs)

    pool_args = [(i, path, task, rescaled_costs) for i, path in enumerate(paths_from_fridge)]
    
    with Pool() as pool:
        results = pool.map(_calculate_utility_for_single_from_path, pool_args)

    utilities, optimal_plant_spots = zip(*results)

    return np.array(utilities), list(optimal_plant_spots)


def group_paths_by_length(paths: List) -> Dict[int, List]:
    """Groups a list of paths by their length."""
    paths_by_len = {}
    for path in paths:
        length = len(path) - 1 if path else 0
        if length not in paths_by_len:
            paths_by_len[length] = []
        paths_by_len[length].append(path)
    return paths_by_len


def calculate_single_audio_utility(
    length_meta: Dict,
    rescaled_cost: float,
    task: PathSamplingTask
) -> float:
    """Calculate utility for a single audio path length pair."""
    
    if task.agent_type == 'sophisticated':
        to_slider_map = getattr(task.config.evidence, 'audio_to_slider_map', {})
        from_slider_map = getattr(task.config.evidence, 'audio_from_slider_map', {})

        eval_steps_to = length_meta['eval_steps_to']
        eval_steps_from = length_meta['eval_steps_from']
        
        slider_to = to_slider_map.get(eval_steps_to, 0)
        slider_from = from_slider_map.get(eval_steps_from, 0)
        
        # Combine audio framing scores by multiplying their likelihood-space equivalents
        # This is a bit of a heuristic to combine sliders.
        # A simple average is slider_to + slider_from / 2
        framing_metric = (slider_to + slider_from) / 2.0
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


def calculate_length_based_probabilities(paths: List, task: PathSamplingTask) -> np.ndarray:
    """Calculate sampling probabilities for a list of paths based on their lengths."""
    if not paths:
        return np.array([])
    costs = np.array([len(p) - 1 for p in paths])
    # Higher utility for shorter paths
    utilities = 1.0 - rescale_costs(costs)
    return utilities_to_probabilities(utilities, task)


def _find_best_framing_spot(
    path: List[Tuple[int, int]],
    agent_id: str,
    visual_slider_map: Dict[Tuple[int, int], float]
) -> Tuple[any, float]:
    """Finds the best coordinate on a path to frame the other agent."""
    
    # We can only plant on the path itself.
    # We also need to filter out non-kitchen/door locations, which is implicitly
    # handled by the fact that the visual_slider_map should only contain valid spots.
    valid_plant_spots = [coord for coord in path if coord in visual_slider_map]

    # Get the slider values for just the valid spots on the path
    slider_values_on_path = {spot: visual_slider_map[spot] for spot in valid_plant_spots}

    # Agent A wants to maximize the slider, Agent B wants to minimize 
    if agent_id == 'A':
        best_spot = max(slider_values_on_path, key=slider_values_on_path.get)
    else: # Agent B
        best_spot = min(slider_values_on_path, key=slider_values_on_path.get)
        
    best_slider_value = slider_values_on_path[best_spot]
    
    return best_spot, best_slider_value


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

    weights = []
    for spot in candidate_spots:
        distance_sq = (spot[0] - optimal_spot[0])**2 + (spot[1] - optimal_spot[1])**2
        weight = np.exp(-distance_sq / (2 * (sigma**2 + 1e-6))) 
        weights.append(weight)

    sum_weights = np.sum(weights)
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
