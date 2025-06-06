"""
Path sampling functionality for simulation agents.

Handles path sampling logic for different evidence types (visual, audio)
and agent behaviors (naive, sophisticated).
"""

import numpy as np
import logging
from typing import Dict, List
from dataclasses import dataclass

from src.core.evidence import get_compressed_audio_from_path
from src.cfg import PathSamplingTask
from .utilities import (
    calculate_audio_utilities,
    calculate_visual_utilities,
    calculate_multimodal_utilities,
    utilities_to_probabilities,
    get_noisy_plant_spot,
    group_paths_by_length,
    calculate_length_based_probabilities,
    calculate_to_fridge_utilities,
    calculate_from_fridge_utilities
)


@dataclass
class SamplingResult:
    """Container for path sampling results"""
    full_sequences: List
    middle_sequences: List 
    return_sequences: List
    chosen_plant_spots: List
    audio_sequences: List
    to_fridge_sequences: List
    full_sequence_lengths: List
    to_fridge_sequence_lengths: List
    middle_sequence_lengths: List
    return_sequence_lengths: List


class PathSampler:
    """Handles path sampling logic for different evidence types and agent behaviors."""
    def __init__(self, world):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.world = world
    
    def sample_paths(self, task: PathSamplingTask) -> Dict:
        """Dispatcher to call the appropriate sampling method based on evidence type."""
        if task.config.evidence.evidence_type == 'visual':
            return self._sample_visual_paths(task)
        elif task.config.evidence.evidence_type == 'audio':
            return self._sample_audio_paths(task)
        elif task.config.evidence.evidence_type == 'multimodal':
            return self._sample_multimodal_paths(task)
        else:
            raise ValueError(f"Unsupported evidence type: {task.config.evidence.evidence_type}")

    def _sample_visual_paths(self, task: PathSamplingTask) -> Dict:
        """Samples paths for visual evidence scenarios."""
        p_start_door, p_door_fridge, p_fridge_door, p_door_start = task.simple_path_sequences
        
        result = SamplingResult([], [], [], [], [], [], [], [], [], [])
        
        if task.agent_type == 'sophisticated':
            # Probability for strategic path (fridge to door) is based on visual utility
            utilities_strategic, optimal_plant_spots = calculate_visual_utilities(task, p_fridge_door)
            probs_fridge_to_door = utilities_to_probabilities(utilities_strategic, task)

            # Probabilities for all other non-strategic segments are based on path length
            probs_start_door = calculate_length_based_probabilities(p_start_door, task)
            probs_door_fridge = calculate_length_based_probabilities(p_door_fridge, task)
            probs_door_start = calculate_length_based_probabilities(p_door_start, task)

        elif task.agent_type == 'naive':
            # For naive, calculate probabilities for the full composite paths based on length
            paths_to_fridge = [p1[:-1] + p2 for p1 in p_start_door for p2 in p_door_fridge]
            probs_to_fridge = calculate_length_based_probabilities(paths_to_fridge, task)
            
            paths_from_fridge = [p3[:-1] + p4 for p3 in p_fridge_door for p4 in p_door_start]
            probs_from_fridge = calculate_length_based_probabilities(paths_from_fridge, task)
            
        # Path sampling
        for _ in range(task.num_sample_paths):
            if task.agent_type == 'naive':
                idx_to = np.random.choice(len(paths_to_fridge), p=probs_to_fridge)
                to_fridge_sequence = paths_to_fridge[idx_to]

                idx_from = np.random.choice(len(paths_from_fridge), p=probs_from_fridge)
                return_sequence = paths_from_fridge[idx_from]
                
                seg_fridge_door = next(p for p in p_fridge_door if tuple(return_sequence[:len(p)]) == tuple(p))
                
            elif task.agent_type == 'sophisticated':
                # Sample each segment according to its calculated probability distribution
                idx_start_door = np.random.choice(len(p_start_door), p=probs_start_door)
                idx_door_to_fridge = np.random.choice(len(p_door_fridge), p=probs_door_fridge)
                idx_fridge_to_door = np.random.choice(len(p_fridge_door), p=probs_fridge_to_door)
                idx_door_to_start = np.random.choice(len(p_door_start), p=probs_door_start)
                
                # Assemble the sequences from the chosen segment indices
                seg_start_door = p_start_door[idx_start_door]
                seg_door_fridge = p_door_fridge[idx_door_to_fridge]
                seg_fridge_door = p_fridge_door[idx_fridge_to_door]
                seg_door_start = p_door_start[idx_door_to_start]
                
                to_fridge_sequence = seg_start_door[:-1] + seg_door_fridge
                return_sequence = seg_fridge_door[:-1] + seg_door_start

            # Save results
            full_sequence = to_fridge_sequence[:-1] + return_sequence
            
            result.full_sequences.append(full_sequence)
            result.to_fridge_sequences.append(to_fridge_sequence)
            result.return_sequences.append(return_sequence)
            result.middle_sequences.append(seg_fridge_door)

            chosen_plant_spot = None
            if task.agent_type == 'sophisticated':
                if optimal_plant_spots[idx_fridge_to_door] is not None:
                    optimal_spot = optimal_plant_spots[idx_fridge_to_door]
                    # Plant crumb
                    if task.config.evidence.crumb_planting_sigma > 0:
                        chosen_plant_spot = get_noisy_plant_spot(optimal_spot, task.config.evidence.crumb_planting_sigma, set(self.world.get_valid_kitchen_crumb_coords()))
                    else:
                        chosen_plant_spot = optimal_spot
            result.chosen_plant_spots.append(chosen_plant_spot)
            
            result.full_sequence_lengths.append(len(full_sequence) - 1)
            result.to_fridge_sequence_lengths.append(len(to_fridge_sequence) - 1)
            result.return_sequence_lengths.append(len(return_sequence) - 1)
            result.middle_sequence_lengths.append(len(seg_fridge_door) - 1)
        
        return result.__dict__

    def _sample_audio_paths(self, task: PathSamplingTask) -> Dict:
        """Samples paths for audio evidence scenarios."""
        p_start_door, p_door_fridge, \
        p_fridge_door, p_door_start = task.simple_path_sequences

        result = SamplingResult([], [], [], [], [], [], [], [], [], [])

        paths_to_fridge = [p1[:-1] + p2 for p1 in p_start_door for p2 in p_door_fridge]
        paths_from_fridge = [p3[:-1] + p4 for p3 in p_fridge_door for p4 in p_door_start]

        # Use the efficient, length-based utility calculation
        length_pair_metadata, probabilities = calculate_audio_utilities(task, paths_to_fridge, paths_from_fridge)

        # Group actual paths by length for quick lookup after sampling a length
        paths_by_len_to = group_paths_by_length(paths_to_fridge)
        paths_by_len_from = group_paths_by_length(paths_from_fridge)

        for _ in range(task.num_sample_paths):
            # Sample a pair of path lengths based on the calculated probabilities
            chosen_idx = np.random.choice(len(length_pair_metadata), p=probabilities)
            chosen_meta = length_pair_metadata[chosen_idx]
            selected_len_to = chosen_meta['eval_steps_to']
            selected_len_from = chosen_meta['eval_steps_from']
            
            # Randomly select an actual path that has the chosen length
            to_fridge_options = paths_by_len_to.get(selected_len_to, [])
            from_fridge_options = paths_by_len_from.get(selected_len_from, [])
            
            if not to_fridge_options or not from_fridge_options:
                self.logger.warning(f"No paths found for sampled lengths to:{selected_len_to}, from:{selected_len_from}. Skipping sample.")
                continue

            to_fridge_sequence = to_fridge_options[np.random.randint(0, len(to_fridge_options))]
            return_sequence = from_fridge_options[np.random.randint(0, len(from_fridge_options))]
            
            # Assemble and save the results
            full_sequence = to_fridge_sequence[:-1] + return_sequence

            seg_fridge_door = next(p for p in p_fridge_door if tuple(return_sequence[:len(p)]) == tuple(p))

            compressed_audio = get_compressed_audio_from_path(task.world, full_sequence)
            
            result.full_sequences.append(full_sequence)
            result.to_fridge_sequences.append(to_fridge_sequence)
            result.return_sequences.append(return_sequence)
            result.middle_sequences.append(seg_fridge_door)
            result.audio_sequences.append(compressed_audio)
            
            result.full_sequence_lengths.append(len(full_sequence) - 1)
            result.to_fridge_sequence_lengths.append(len(to_fridge_sequence) - 1)
            result.return_sequence_lengths.append(len(return_sequence) - 1)
            result.middle_sequence_lengths.append(len(seg_fridge_door) - 1)
            
        return result.__dict__

    def _sample_multimodal_paths(self, task: PathSamplingTask) -> Dict:
        """Samples paths for multimodal evidence scenarios."""
        p_start_door, p_door_fridge, \
        p_fridge_door, p_door_start = task.simple_path_sequences

        result = SamplingResult([], [], [], [], [], [], [], [], [], [])
        
        paths_to_fridge = [p1[:-1] + p2 for p1 in p_start_door for p2 in p_door_fridge]
        paths_from_fridge = [p3[:-1] + p4 for p3 in p_fridge_door for p4 in p_door_start]

        if task.agent_type == 'naive':
            probs_to_fridge = calculate_length_based_probabilities(paths_to_fridge, task)
            probs_from_fridge = calculate_length_based_probabilities(paths_from_fridge, task)

        elif task.agent_type == 'sophisticated':
            # Calculate probabilities for each leg of the journey independently            
            # Utilities and probabilities for path to the fridge
            utilities_to = calculate_to_fridge_utilities(task, paths_to_fridge)
            probs_to_fridge = utilities_to_probabilities(utilities_to, task)

            # Utilities and probabilities for path from the fridge
            utilities_from, optimal_spots_from = calculate_from_fridge_utilities(task, paths_from_fridge)
            probs_from_fridge = utilities_to_probabilities(utilities_from, task)

        # Path sampling
        for _ in range(task.num_sample_paths):
            chosen_plant_spot = None
            if task.agent_type == 'naive':
                idx_to = np.random.choice(len(paths_to_fridge), p=probs_to_fridge)
                to_fridge_sequence = paths_to_fridge[idx_to]

                idx_from = np.random.choice(len(paths_from_fridge), p=probs_from_fridge)
                return_sequence = paths_from_fridge[idx_from]
            
            elif task.agent_type == 'sophisticated':
                # Sample each leg independently
                idx_to = np.random.choice(len(paths_to_fridge), p=probs_to_fridge)
                to_fridge_sequence = paths_to_fridge[idx_to]

                idx_from = np.random.choice(len(paths_from_fridge), p=probs_from_fridge)
                return_sequence = paths_from_fridge[idx_from]
                
                optimal_spot = optimal_spots_from[idx_from]
                if optimal_spot is not None:
                    if task.config.evidence.crumb_planting_sigma > 0:
                        chosen_plant_spot = get_noisy_plant_spot(optimal_spot, task.config.evidence.crumb_planting_sigma, set(self.world.get_valid_kitchen_crumb_coords()))
                    else:
                        chosen_plant_spot = optimal_spot

            # Assemble and save the results
            full_sequence = to_fridge_sequence[:-1] + return_sequence
            seg_fridge_door = next(p for p in p_fridge_door if tuple(return_sequence[:len(p)]) == tuple(p))
            compressed_audio = get_compressed_audio_from_path(task.world, full_sequence)
            
            result.full_sequences.append(full_sequence)
            result.to_fridge_sequences.append(to_fridge_sequence)
            result.return_sequences.append(return_sequence)
            result.middle_sequences.append(seg_fridge_door)
            result.audio_sequences.append(compressed_audio)
            result.chosen_plant_spots.append(chosen_plant_spot)
            
            result.full_sequence_lengths.append(len(full_sequence) - 1)
            result.to_fridge_sequence_lengths.append(len(to_fridge_sequence) - 1)
            result.return_sequence_lengths.append(len(return_sequence) - 1)
            result.middle_sequence_lengths.append(len(seg_fridge_door) - 1)
            
        return result.__dict__
    