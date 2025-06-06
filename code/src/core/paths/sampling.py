"""
Path sampling functionality for simulation agents.

Handles path sampling logic for different evidence types (visual, audio)
and agent behaviors (naive, sophisticated).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.core.evidence import get_compressed_audio_from_path
from src.cfg import PathSamplingTask
from .utilities import (
    calculate_audio_utilities,
    group_paths_by_length,
    sample_paths_with_lengths,
    utilities_to_probabilities,
    calculate_optimal_plant_spot_and_slider,
    get_noisy_plant_spot,
    is_valid_audio_sequence
)


@dataclass
class SamplingResult:
    """Container for path sampling results"""
    full_sequences: List
    middle_sequences: List
    chosen_plant_spots: List
    audio_sequences: List
    to_fridge_sequences: List
    full_sequence_lengths: List
    to_fridge_sequence_lengths: List
    middle_sequence_lengths: List


class PathSampler:
    """Handles path sampling logic for different evidence types and agent behaviors."""
    def __init__(self, world):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.audio_framing_cache = {}
        self.world = world
    

    def sample_paths(self, task: PathSamplingTask) -> Dict:
        """Main entry point for path sampling."""
        if task.config.evidence.evidence_type == 'visual':
            return self._sample_visual_paths(task)
        elif task.config.evidence.evidence_type == 'audio':
            return self._sample_audio_paths(task)
        else:
            raise ValueError(f"Unsupported evidence type: {task.config.evidence.evidence_type}")
    

    def _sample_visual_paths(self, task: PathSamplingTask) -> Dict:
        """Sample paths for visual evidence."""
        
        sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = task.simple_path_sequences
        
        # Calculate utilities and probabilities for P2 (Fridge->Door)
        utilities, optimal_plant_spots = self._calculate_visual_utilities(task, sequences_p2)
        
        # Handle case where no utilities were calculated
        if len(utilities) == 0:
            self.logger.error(f"No valid utilities calculated for agent {task.agent_id}")
            return SamplingResult([], [], [], [], [], [], [], []).__dict__
        
        probabilities = utilities_to_probabilities(utilities, task)
        
        # Sample paths
        result = SamplingResult([], [], [], [], [], [], [], [])
        
        for _ in range(task.num_sample_paths):
            # Randomly select path segments
            idx1 = np.random.randint(0, len(sequences_p1))
            idx3 = np.random.randint(0, len(sequences_p3))
            idx2 = np.random.choice(len(sequences_p2), p=probabilities)
            
            p1_seq = sequences_p1[idx1]  # Start -> Fridge
            p2_seq = sequences_p2[idx2]  # Fridge -> Door
            p3_seq = sequences_p3[idx3]  # Door -> Start
            
            # Combine sequences
            full_sequence = p1_seq[:-1] + p2_seq[:-1] + p3_seq
            
            result.full_sequences.append(full_sequence)
            result.middle_sequences.append(p2_seq)
            result.to_fridge_sequences.append(p1_seq)
            
            # Calculate lengths
            result.full_sequence_lengths.append(len(full_sequence) - 1 if full_sequence else 0)
            result.to_fridge_sequence_lengths.append(len(p1_seq) - 1 if p1_seq else 0)
            result.middle_sequence_lengths.append(len(p2_seq) - 1 if p2_seq else 0)
            
            # Handle plant spots for sophisticated agents
            chosen_plant_spot = None
            valid_plant_spots = set(self.world.get_valid_kitchen_crumb_coords())

            if task.agent_type == 'sophisticated' and optimal_plant_spots[idx2] is not None:
                optimal_spot = optimal_plant_spots[idx2]
                if task.config.evidence.crumb_planting_sigma > 0:
                    # add noise to crumb planting execution
                    chosen_plant_spot = get_noisy_plant_spot(
                        optimal_spot, 
                        task.config.evidence.crumb_planting_sigma, 
                        valid_plant_spots
                    )
                else:
                    chosen_plant_spot = optimal_spot
            result.chosen_plant_spots.append(chosen_plant_spot)
        
        return result.__dict__
    

    def _sample_audio_paths(self, task: PathSamplingTask) -> Dict:
        """Sample paths for audio evidence."""
        
        sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = task.simple_path_sequences
        candidate_paths_to_fridge = sequences_p1
        candidate_paths_from_fridge = sequences_fridge_to_start
        
        # Group paths by length and calculate utilities
        length_pair_metadata, probabilities = calculate_audio_utilities(task, candidate_paths_to_fridge, candidate_paths_from_fridge)
        
        # Group paths for efficient sampling
        paths_by_len_to, paths_by_len_from = group_paths_by_length(
            candidate_paths_to_fridge, candidate_paths_from_fridge
        )
        
        # Sample paths
        result = SamplingResult([], [], [], [], [], [], [], [])
        
        for _ in range(task.num_sample_paths):
            # Choose length pair based on utility
            chosen_idx = np.random.choice(len(length_pair_metadata), p=probabilities)
            chosen_meta = length_pair_metadata[chosen_idx]
            
            selected_len_to = chosen_meta['eval_steps_to']
            selected_len_from = chosen_meta['eval_steps_from']
            
            # Sample actual paths with chosen lengths
            p_to_seq, p_from_seq = sample_paths_with_lengths(
                paths_by_len_to, paths_by_len_from, selected_len_to, selected_len_from
            )
            
            # Combine paths and create audio signature
            full_sequence = p_to_seq[:-1] + p_from_seq if p_to_seq and p_from_seq else []
            compressed_audio = get_compressed_audio_from_path(task.world, full_sequence)
            
            # Validate audio format
            if not is_valid_audio_sequence(compressed_audio):
                self.logger.warning(f"Agent {task.agent_id} ({task.agent_type}) AUDIO: Malformed compressed audio. Storing as None.")
                compressed_audio = None
            
            # Store results
            result.full_sequences.append(full_sequence)
            result.to_fridge_sequences.append(p_to_seq)
            result.middle_sequences.append(p_from_seq)  # For audio, middle = from_fridge
            result.audio_sequences.append(compressed_audio)
            
            # Calculate lengths
            result.full_sequence_lengths.append(len(full_sequence) - 1 if full_sequence else 0)
            result.to_fridge_sequence_lengths.append(len(p_to_seq) - 1 if p_to_seq else 0)
            result.middle_sequence_lengths.append(len(p_from_seq) - 1 if p_from_seq else 0)
        
        return result.__dict__
    

    def _calculate_visual_utilities(
        self,
        task: PathSamplingTask,
        sequences_p2: List
    ) -> Tuple[np.ndarray, List]:
        """Calculate utilities for visual path segments (P2: Fridge->Door)."""
        
        # Calculate length-based utilities
        middle_path_lengths = np.array([len(seq) for seq in sequences_p2])
        
        min_len, max_len = np.min(middle_path_lengths), np.max(middle_path_lengths)
        rescaled_lengths = np.zeros_like(middle_path_lengths, dtype=float)
        if max_len > min_len:
            rescaled_lengths = (middle_path_lengths - min_len) / (max_len - min_len)
        
        optimal_plant_spots = [None] * len(sequences_p2)
        utilities = []
        
        if task.agent_type == 'sophisticated':
            # Calculate optimal plant spots and framing utilities
            fridge_access_point = task.world.get_fridge_access_point()
            for idx, p2_seq in enumerate(sequences_p2):
                optimal_spot, best_slider = calculate_optimal_plant_spot_and_slider(
                    task.world, task.agent_id, p2_seq, fridge_access_point, task.config
                )
                optimal_plant_spots[idx] = optimal_spot
                
                path_framing_metric_scaled = best_slider / 100.0  # [-0.5, 0.5]
                utility_factor = task.config.sampling.cost_weight * (1 - rescaled_lengths[idx])
                
                if task.agent_id == 'A':
                    utility = utility_factor + (1 - task.config.sampling.cost_weight) * path_framing_metric_scaled
                else:
                    utility = utility_factor - (1 - task.config.sampling.cost_weight) * path_framing_metric_scaled
                utilities.append(utility)
        
        elif task.agent_type in ['naive', 'uniform']:
            # Simple cost-based utilities
            for l_rescaled in rescaled_lengths:
                utilities.append(task.config.sampling.cost_weight * (1 - l_rescaled))
        
        return np.array(utilities), optimal_plant_spots 
    