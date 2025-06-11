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
from src.cfg import PathSamplingTask


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
        for idx, p2_seq in enumerate(sequences_p2):
            # First find the best framing spot based on slider map
            optimal_spot, best_slider = _find_best_framing_spot(p2_seq, task.agent_id, visual_slider_map)
            # print(f"Optimal spot: {optimal_spot}")
            # print(f"Best slider: {best_slider}")
            
            # Then get a noisy version of the optimal spot for actual planting
            if optimal_spot is not None:
                valid_plant_spots = set(visual_slider_map.keys())
                noisy_plant_spot = get_noisy_plant_spot(optimal_spot, task.config.evidence.crumb_planting_sigma, valid_plant_spots)
                optimal_plant_spots[idx] = noisy_plant_spot
            else:
                optimal_plant_spots[idx] = None
            
            path_framing_metric_scaled = best_slider / 100.0
            
            # Path utility
            # Agent A: 1 - path_length + k * detective_pred (higher detective_pred -> accuse B)
            # Agent B: 1 - path_length - k * detective_pred (higher detective_pred -> accuse A)
            if task.agent_id == 'A':
                # print(f"Agent A path length: {rescaled_lengths[idx]}")
                # print(f"Agent A path framing metric: {path_framing_metric_scaled}")
                # print(f"Agent A cost weight: {task.config.sampling.cost_weight}")
                utility = (1 - rescaled_lengths[idx]) + task.config.sampling.cost_weight * path_framing_metric_scaled
                # print(f"Agent A utility: {utility}")
                # breakpoint()
            else:
                utility = (1 - rescaled_lengths[idx]) - task.config.sampling.cost_weight * path_framing_metric_scaled
                # print(f"Agent B utility: {utility}")
                
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


def calculate_to_fridge_utilities(task: PathSamplingTask, paths_to_fridge: List) -> np.ndarray:
    """Calculate utilities for 'to fridge' paths using pre-computed likelihoods."""
    audio_to_lik_map = getattr(task.config.evidence, 'audio_to_likelihood_map', {})

    costs = np.array([len(p) - 1 for p in paths_to_fridge])
    rescaled_costs = rescale_costs(costs)
    
    utilities = []
    for i, path in enumerate(paths_to_fridge):
        path_len = len(path) - 1
        
        # Get likelihoods and compute framing metric
        lik_A_audio, lik_B_audio = audio_to_lik_map.get(path_len, (0.0, 0.0))
        framing_metric = normalized_slider_prediction(lik_A_audio, lik_B_audio) / 100.0
        
        # (1 - path_length) + k * detective_pred (agent A), (1 - path_length) - k * detective_pred (agent B)
        utility = (1 - rescaled_costs[i]) + task.config.sampling.cost_weight * framing_metric if task.agent_id == 'A' \
            else (1 - rescaled_costs[i]) - task.config.sampling.cost_weight * framing_metric
        utilities.append(utility)

    return np.array(utilities)


def _calculate_utility_for_single_from_path(args: Tuple) -> Tuple[float, any]:
    """
    Worker function to calculate utility for a single 'from' path for multimodal agents.
    """
    i, p_from, task, rescaled_costs = args
    
    config = task.config
    agent_id = task.agent_id
    
    # Get pre-computed likelihood maps
    visual_lik_map_A = getattr(config.evidence, 'naive_A_visual_likelihoods_map', {})
    visual_lik_map_B = getattr(config.evidence, 'naive_B_visual_likelihoods_map', {})
    audio_from_lik_map = getattr(config.evidence, 'audio_from_likelihood_map', {})

    # Find the best plant spot based on the visual slider map, as this is the agent's goal
    visual_slider_map = getattr(config.evidence, 'visual_slider_map', {})
    optimal_spot, _ = _find_best_framing_spot(p_from, agent_id, visual_slider_map)

    # Get the likelihoods for that optimal spot
    lik_A_visual = visual_lik_map_A.get(optimal_spot, 0)
    lik_B_visual = visual_lik_map_B.get(optimal_spot, 0)
    
    # Get audio likelihoods for the path's length
    len_from = len(p_from) - 1
    lik_A_audio, lik_B_audio = audio_from_lik_map.get(len_from, (0.0, 0.0))
    
    # Combine likelihoods by multiplying
    total_multimodal_lik_A = lik_A_visual * lik_A_audio
    total_multimodal_lik_B = lik_B_visual * lik_B_audio
    
    # Calculate the final slider prediction from the combined likelihoods
    framing_metric = normalized_slider_prediction(total_multimodal_lik_A, total_multimodal_lik_B) / 100.0
    
    # (1 - path_length) + k * detective_pred (agent A), (1 - path_length) - k * detective_pred (agent B)
    utility = (1 - rescaled_costs[i]) + config.sampling.cost_weight * framing_metric if agent_id == 'A' \
        else (1 - rescaled_costs[i]) - config.sampling.cost_weight * framing_metric
        
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
        # Get likelihood maps instead of slider maps
        audio_to_lik_map = getattr(task.config.evidence, 'audio_to_likelihood_map', {})
        audio_from_lik_map = getattr(task.config.evidence, 'audio_from_likelihood_map', {})

        eval_steps_to = length_meta['eval_steps_to']
        eval_steps_from = length_meta['eval_steps_from']
        
        # Get likelihoods for both path segments
        lik_A_to, lik_B_to = audio_to_lik_map.get(eval_steps_to, (0.0, 0.0))
        lik_A_from, lik_B_from = audio_from_lik_map.get(eval_steps_from, (0.0, 0.0))
        
        # Combine likelihoods by multiplying
        total_lik_A = lik_A_to * lik_A_from
        total_lik_B = lik_B_to * lik_B_from
        
        # Convert combined likelihoods back to slider prediction
        framing_metric = normalized_slider_prediction(total_lik_A, total_lik_B) / 100.0
        
        # (path_length) + k * detective_pred (agent A), (path_length) - k * detective_pred (agent B)
        utility = rescaled_cost + task.config.sampling.cost_weight * framing_metric if task.agent_id == 'A' \
            else rescaled_cost - task.config.sampling.cost_weight * framing_metric
        
        return utility
    
    elif task.agent_type in ['naive', 'uniform']:
        return rescaled_cost


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
    valid_plant_spots = [coord for coord in path if coord in visual_slider_map]

    # Get the slider values for valid spots on path
    slider_values_on_path = {spot: visual_slider_map[spot] for spot in valid_plant_spots}

    # Agent A wants to maximize the slider, Agent B wants to minimize 
    best_spot = max(slider_values_on_path, key=slider_values_on_path.get) if agent_id == 'A' \
        else min(slider_values_on_path, key=slider_values_on_path.get)
        
    best_slider_value = slider_values_on_path[best_spot]
    return best_spot, best_slider_value


def get_noisy_plant_spot(optimal_spot: Tuple[int, int], 
                         sigma: float, 
                         valid_plant_spots: Set[Tuple[int, int]]) -> Tuple[int, int]:
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
