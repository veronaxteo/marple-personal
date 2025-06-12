"""
Path sampling functionality for simulation agents.

Handles path sampling logic for different evidence types (visual, audio, multimodal)
and agent behaviors (naive, sophisticated).
"""

import numpy as np
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import logging
from src.core.evidence import get_compressed_audio_from_path
from src.cfg import PathSamplingTask
from .utilities import (
    calculate_visual_utilities,
    utilities_to_probabilities,
    get_noisy_plant_spot,
    group_paths_by_length,
    calculate_length_based_probabilities,
    calculate_to_fridge_utilities,
    calculate_from_fridge_utilities,
    calculate_audio_utilities
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
        self.utility_cache = {}

    def sample_paths(self, task: PathSamplingTask) -> Dict:
        """Dispatcher to call the appropriate sampling method based on evidence type."""
        if task.agent_type == 'naive':
            return self._sample_paths_naive(task)

        elif task.agent_type == 'sophisticated':
            cache_key = (task.agent_id, task.agent_type, task.config.evidence.evidence_type)
            if cache_key in self.utility_cache:
                self.logger.info(f"Using cached utilities for {cache_key}")
                cached_data = self.utility_cache[cache_key]
            else:
                self.logger.info(f"No cache found. Computing utilities for {cache_key}")
                if task.config.evidence.evidence_type == 'visual':
                    cached_data = self._compute_visual_utilities_sophisticated(task)
                elif task.config.evidence.evidence_type == 'audio':
                    cached_data = self._compute_audio_utilities_sophisticated(task)
                elif task.config.evidence.evidence_type == 'multimodal':
                    cached_data = self._compute_multimodal_utilities_sophisticated(task)
                else:
                    raise ValueError(f"Unsupported evidence type: {task.config.evidence.evidence_type}")
                self.utility_cache[cache_key] = cached_data

            return self._sample_from_cached_utilities(task, cached_data)
        
        raise ValueError(f"Unsupported agent type: {task.agent_type}")

    def _sample_paths_naive(self, task: PathSamplingTask) -> Dict:
        """Naive sampling is always based on the full journey's path length."""
        p_start_door, p_door_fridge, p_fridge_door, p_door_start = task.simple_path_sequences
        # print(f"p_start_door: {p_start_door}")
        # breakpoint()

        paths_to_fridge = [p1[:-1] + p2 for p1 in p_start_door for p2 in p_door_fridge]
        paths_from_fridge = [p3[:-1] + p4 for p3 in p_fridge_door for p4 in p_door_start]
        
        probs_to_fridge = calculate_length_based_probabilities(paths_to_fridge, task)
        probs_from_fridge = calculate_length_based_probabilities(paths_from_fridge, task)
        
        result = SamplingResult([], [], [], [], [], [], [], [], [], [])
        for _ in range(task.num_sample_paths):
            idx_to = np.random.choice(len(paths_to_fridge), p=probs_to_fridge)
            to_fridge_sequence = paths_to_fridge[idx_to]

            idx_from = np.random.choice(len(paths_from_fridge), p=probs_from_fridge)
            return_sequence = paths_from_fridge[idx_from]
            
            self._assemble_and_save_results(result, task, to_fridge_sequence, return_sequence, None)
            
        return result.__dict__


    def _compute_visual_utilities_sophisticated(self, task: PathSamplingTask) -> Dict:
        """Computes and caches utilities for sophisticated visual agents."""
        _, p_door_fridge, p_fridge_door, _ = task.simple_path_sequences
        
        # Strategic part: visual utility for the return path
        utilities_strategic, optimal_plant_spots = calculate_visual_utilities(task, p_fridge_door)
        probs_fridge_to_door = utilities_to_probabilities(utilities_strategic, task)
        
        # Length-based probabilities for the path to the fridge
        probs_door_fridge = calculate_length_based_probabilities(p_door_fridge, task)
        
        return {
            'type': 'visual',
            'probs_fridge_to_door': probs_fridge_to_door,
            'optimal_plant_spots': optimal_plant_spots,
            'probs_door_fridge': probs_door_fridge,
            'p_door_fridge': p_door_fridge,
            'p_fridge_door': p_fridge_door,
        }

    def _compute_audio_utilities_sophisticated(self, task: PathSamplingTask) -> Dict:
        """Computes and caches utilities for sophisticated audio agents."""
        _, p_door_fridge, p_fridge_door, _ = task.simple_path_sequences
        
        length_pair_metadata, probabilities = calculate_audio_utilities(task, p_door_fridge, p_fridge_door)
        
        return {
            'type': 'audio',
            'length_pair_metadata': length_pair_metadata,
            'probabilities': probabilities,
            'p_door_fridge': p_door_fridge,
            'p_fridge_door': p_fridge_door
        }

    def _compute_multimodal_utilities_sophisticated(self, task: PathSamplingTask) -> Dict:
        """Computes and caches utilities for sophisticated multimodal agents."""
        _, p_door_fridge, p_fridge_door, _ = task.simple_path_sequences

        self.logger.info(f"Computing utilities for {len(p_door_fridge)} paths to fridge.")
        utilities_to = calculate_to_fridge_utilities(task, p_door_fridge)
        probs_to_fridge = utilities_to_probabilities(utilities_to, task)
        self.logger.info(f"Finished computing utilities for {len(p_door_fridge)} paths to fridge.")

        self.logger.info(f"Computing utilities for {len(p_fridge_door)} paths from fridge.")
        utilities_from, optimal_spots_from = calculate_from_fridge_utilities(task, p_fridge_door)
        probs_from_fridge = utilities_to_probabilities(utilities_from, task)
        self.logger.info(f"Finished computing utilities for {len(p_fridge_door)} paths from fridge.")

        return {
            'type': 'multimodal',
            'probs_to_fridge': probs_to_fridge,
            'probs_from_fridge': probs_from_fridge,
            'optimal_spots_from': optimal_spots_from,
            'p_door_fridge': p_door_fridge,
            'p_fridge_door': p_fridge_door
        }

    def _sample_from_cached_utilities(self, task: PathSamplingTask, cached_data: Dict) -> Dict:
        """Generic sampling loop that uses cached utility/probability data."""
        p_start_door, _, _, p_door_start = task.simple_path_sequences
        
        probs_start_door = calculate_length_based_probabilities(p_start_door, task)
        probs_door_start = calculate_length_based_probabilities(p_door_start, task)
        
        result = SamplingResult([], [], [], [], [], [], [], [], [], [])
        for _ in range(task.num_sample_paths):
            chosen_plant_spot = None
            
            if cached_data['type'] == 'visual':
                idx_to = np.random.choice(len(cached_data['p_door_fridge']), p=cached_data['probs_door_fridge'])
                seg_door_fridge = cached_data['p_door_fridge'][idx_to]
                
                idx_from = np.random.choice(len(cached_data['p_fridge_door']), p=cached_data['probs_fridge_to_door'])
                seg_fridge_door = cached_data['p_fridge_door'][idx_from]
                
                optimal_spot = cached_data['optimal_plant_spots'][idx_from]
                chosen_plant_spot = self._get_final_plant_spot(optimal_spot, task)
            
            elif cached_data['type'] == 'audio':
                paths_by_len_to = group_paths_by_length(cached_data['p_door_fridge'])
                paths_by_len_from = group_paths_by_length(cached_data['p_fridge_door'])
                idx = np.random.choice(len(cached_data['length_pair_metadata']), p=cached_data['probabilities'])
                meta = cached_data['length_pair_metadata'][idx]
                
                to_options = paths_by_len_to.get(meta['eval_steps_to'], [])
                from_options = paths_by_len_from.get(meta['eval_steps_from'], [])

                seg_door_fridge = to_options[np.random.randint(0, len(to_options))]
                seg_fridge_door = from_options[np.random.randint(0, len(from_options))]

            elif cached_data['type'] == 'multimodal':
                idx_to = np.random.choice(len(cached_data['p_door_fridge']), p=cached_data['probs_to_fridge'])
                seg_door_fridge = cached_data['p_door_fridge'][idx_to]
                
                idx_from = np.random.choice(len(cached_data['p_fridge_door']), p=cached_data['probs_from_fridge'])
                seg_fridge_door = cached_data['p_fridge_door'][idx_from]

                optimal_spot = cached_data['optimal_spots_from'][idx_from]
                chosen_plant_spot = self._get_final_plant_spot(optimal_spot, task)

            # Assemble the full path
            chosen_start_door = p_start_door[np.random.choice(len(p_start_door), p=probs_start_door)]
            chosen_door_start = p_door_start[np.random.choice(len(p_door_start), p=probs_door_start)]
            # chosen_start_door = p_start_door[np.random.randint(0, len(p_start_door))]
            # chosen_door_start = p_door_start[np.random.randint(0, len(p_door_start))]
            
            to_fridge_sequence = chosen_start_door[:-1] + seg_door_fridge
            return_sequence = seg_fridge_door[:-1] + chosen_door_start
            
            self._assemble_and_save_results(result, task, to_fridge_sequence, return_sequence, chosen_plant_spot)
            
        return result.__dict__

    def _get_final_plant_spot(self, optimal_spot: Any, task: PathSamplingTask) -> Any:
        """Helper to get a noisy plant spot from an optimal one."""
        if optimal_spot is not None:
            sigma = task.config.evidence.crumb_planting_sigma
            if sigma > 0:
                valid_coords = set(self.world.get_valid_kitchen_crumb_coords())
                return get_noisy_plant_spot(optimal_spot, sigma, valid_coords)
            return optimal_spot
        return None

    def _assemble_and_save_results(self, result: SamplingResult, task: PathSamplingTask, to_fridge_sequence: List, return_sequence: List, chosen_plant_spot: Any):
        """Helper to assemble final sequences and append all results."""
        full_sequence = to_fridge_sequence[:-1] + return_sequence
        
        p_fridge_door = task.simple_path_sequences[2]
        seg_fridge_door = next((p for p in p_fridge_door if tuple(return_sequence[-len(p):]) == tuple(p)), [])

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
        result.middle_sequence_lengths.append(len(seg_fridge_door) - 1 if seg_fridge_door else 0)
