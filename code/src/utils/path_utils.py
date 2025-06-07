"""
Utilities for sophisticated agent strategic calculations.

Provides functions to pre-compute "slider maps" based on a naive detective's models.
These maps can be used by sophisticated agents to quickly evaluate the framing
potential of their actions without re-running expensive calculations.
"""
import numpy as np
from typing import Dict, Tuple

from src.cfg import SimulationConfig
from src.utils.math_utils import normalized_slider_prediction
from src.core.evidence.audio import single_segment_audio_likelihood


def create_visual_slider_map(
    visual_model_A: Dict[Tuple[int, int], float], 
    visual_model_B: Dict[Tuple[int, int], float]
) -> Dict[Tuple[int, int], float]:
    """
    Creates a map from a coordinate to its slider prediction value.

    Args:
        visual_model_A: A map of {coord: likelihood} for agent A.
        visual_model_B: A map of {coord: likelihood} for agent B.

    Returns:
        A map of {coord: slider_value}, where slider_value is from -100 (A) to 100 (B).
    """
    all_coords = set(visual_model_A.keys()) | set(visual_model_B.keys())
    
    slider_map = {
        coord: normalized_slider_prediction(
            visual_model_A.get(coord, 0),
            visual_model_B.get(coord, 0)
        ) for coord in all_coords
    }
    return slider_map


def _calculate_single_segment_audio_likelihoods(
    eval_steps: int, 
    direction: str, 
    audio_model_A: Tuple[list, list],
    audio_model_B: Tuple[list, list],
    config: SimulationConfig
) -> Tuple[float, float]:
    """Computes audio likelihoods for a single path segment (to or from)."""
    if direction == 'to':
        naive_A_steps = audio_model_A[0]
        naive_B_steps = audio_model_B[0]
    elif direction == 'from':
        naive_A_steps = audio_model_A[1]
        naive_B_steps = audio_model_B[1]
    else:
        raise ValueError(f"Invalid direction for audio likelihood calculation: {direction}")

    sigma_factor = config.evidence.audio_similarity_sigma
    
    likelihood_A = sum(single_segment_audio_likelihood(eval_steps, step, sigma_factor) 
                       for step in naive_A_steps) / len(naive_A_steps) if naive_A_steps else 0
    likelihood_B = sum(single_segment_audio_likelihood(eval_steps, step, sigma_factor) 
                       for step in naive_B_steps) / len(naive_B_steps) if naive_B_steps else 0
    
    return likelihood_A, likelihood_B


def create_audio_slider_maps(
    audio_model_A: Tuple[list, list], 
    audio_model_B: Tuple[list, list],
    config: SimulationConfig
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Creates maps from a path length to its slider prediction value for to/from segments.

    Args:
        audio_model_A: A tuple of ([to_steps], [from_steps]) for agent A.
        audio_model_B: A tuple of ([to_steps], [from_steps]) for agent B.
        config: The simulation configuration, used for max_steps.

    Returns:
        A tuple of (to_slider_map, from_slider_map), where each map is
        {path_length: slider_value}.
    """
    to_slider_map = {}
    from_slider_map = {}
    max_len = config.sampling.max_steps
    
    # Evaluate for all possible path lengths from 1 to max_steps
    for length in range(1, max_len + 1):
        # "To" segment
        lik_A_to, lik_B_to = _calculate_single_segment_audio_likelihoods(length, 'to', audio_model_A, audio_model_B, config)
        to_slider_map[length] = normalized_slider_prediction(lik_A_to, lik_B_to)
        
        # "From" segment
        lik_A_from, lik_B_from = _calculate_single_segment_audio_likelihoods(length, 'from', audio_model_A, audio_model_B, config)
        from_slider_map[length] = normalized_slider_prediction(lik_A_from, lik_B_from)
        
    return to_slider_map, from_slider_map 


def create_audio_likelihood_maps(
    audio_model_A: Tuple[list, list], 
    audio_model_B: Tuple[list, list],
    config: SimulationConfig
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    """
    Creates maps from a path length to its raw (A, B) likelihood pair.

    Args:
        audio_model_A: A tuple of ([to_steps], [from_steps]) for agent A.
        audio_model_B: A tuple of ([to_steps], [from_steps]) for agent B.
        config: The simulation configuration, used for max_steps.

    Returns:
        A tuple of (to_likelihood_map, from_likelihood_map), where each map is
        {path_length: (likelihood_A, likelihood_B)}.
    """
    to_likelihood_map = {}
    from_likelihood_map = {}
    max_len = config.sampling.max_steps
    
    # Evaluate for all possible path lengths from 1 to max_steps
    for length in range(1, max_len + 1):
        # "To" segment
        lik_A_to, lik_B_to = _calculate_single_segment_audio_likelihoods(length, 'to', audio_model_A, audio_model_B, config)
        to_likelihood_map[length] = (lik_A_to, lik_B_to)
        
        # "From" segment
        lik_A_from, lik_B_from = _calculate_single_segment_audio_likelihoods(length, 'from', audio_model_A, audio_model_B, config)
        from_likelihood_map[length] = (lik_A_from, lik_B_from)
        
    return to_likelihood_map, from_likelihood_map 
