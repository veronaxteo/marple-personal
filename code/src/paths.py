import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.utils.math_utils import softmax_list_vals, normalized_slider_prediction
from src.evidence import get_compressed_audio_from_path, single_segment_audio_likelihood
from src.config import SimulationConfig, PathSamplingTask


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
    """Handles path sampling logic for different evidence types and agent behaviors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.audio_framing_cache = {}
    
    def sample_paths(self, task: PathSamplingTask) -> Dict:
        """Main entry point for path sampling with new config system"""
        
        if task.config.evidence.evidence_type == 'visual':
            return self._sample_visual_paths(task)
        elif task.config.evidence.evidence_type == 'audio':
            return self._sample_audio_paths(task)
        else:
            raise ValueError(f"Unsupported evidence type: {task.config.evidence.evidence_type}")
    
    def _sample_visual_paths(self, task: PathSamplingTask) -> Dict:
        """Sample paths for visual evidence using new config system"""
        
        sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = task.simple_path_sequences
        
        # Calculate utilities and probabilities for P2 (Fridge->Door)
        utilities, optimal_plant_spots = self._calculate_visual_utilities(task, sequences_p2)
        probabilities = self._utilities_to_probabilities(utilities, task)
        
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
            if task.agent_type == 'sophisticated' and optimal_plant_spots[idx2] is not None:
                optimal_spot = optimal_plant_spots[idx2]
                if task.config.sampling.noisy_planting_sigma > 0:
                    chosen_plant_spot = self._get_noisy_plant_spot(
                        task.world, optimal_spot, task.config.sampling.noisy_planting_sigma
                    )
                else:
                    chosen_plant_spot = optimal_spot
            result.chosen_plant_spots.append(chosen_plant_spot)
        
        return result.__dict__
    
    def _sample_audio_paths(self, task: PathSamplingTask) -> Dict:
        """Sample paths for audio evidence using new config system"""
        
        sequences_p1, sequences_p2, sequences_p3, sequences_fridge_to_start = task.simple_path_sequences
        candidate_paths_to_fridge = sequences_p1
        candidate_paths_from_fridge = sequences_fridge_to_start
        
        # Group paths by length and calculate utilities
        length_pair_metadata, probabilities = self._calculate_audio_utilities(task, candidate_paths_to_fridge, candidate_paths_from_fridge)
        
        # Group paths for efficient sampling
        paths_by_len_to, paths_by_len_from = self._group_paths_by_length(
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
            p_to_seq, p_from_seq = self._sample_paths_with_lengths(
                paths_by_len_to, paths_by_len_from, selected_len_to, selected_len_from
            )
            
            # Combine paths and create audio signature
            full_sequence = p_to_seq[:-1] + p_from_seq if p_to_seq and p_from_seq else []
            compressed_audio = get_compressed_audio_from_path(task.world, full_sequence)
            
            # Validate audio format
            if not self._is_valid_audio_sequence(compressed_audio):
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
        """Calculate utilities for visual path segments (P2: Fridge->Door)"""
        
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
                optimal_spot, best_slider = self._calculate_optimal_plant_spot_and_slider(
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
            # Simple cost-based utilities
            for l_rescaled in rescaled_lengths:
                utilities.append(task.config.sampling.cost_weight * (1 - l_rescaled))
        
        return np.array(utilities), optimal_plant_spots
    
    def _calculate_audio_utilities(
        self,
        task: PathSamplingTask,
        candidate_paths_to_fridge: List,
        candidate_paths_from_fridge: List
    ) -> Tuple[List[Dict], np.ndarray]:
        """Calculate utilities for audio path length pairs"""
        
        # Group paths by length
        paths_by_len_to, paths_by_len_from = self._group_paths_by_length(
            candidate_paths_to_fridge, candidate_paths_from_fridge
        )
        
        # Create length pair metadata
        length_pair_metadata = []
        unique_lengths_to = sorted(paths_by_len_to.keys())
        unique_lengths_from = sorted(paths_by_len_from.keys())
        
        for eval_steps_to in unique_lengths_to:
            for eval_steps_from in unique_lengths_from:
                if eval_steps_to in paths_by_len_to and eval_steps_from in paths_by_len_from:
                    length_pair_metadata.append({
                        'eval_steps_to': eval_steps_to,
                        'eval_steps_from': eval_steps_from,
                        'cost': eval_steps_to + eval_steps_from
                    })
        
        if not length_pair_metadata:
            return [], np.array([])
        
        # Calculate utilities
        costs = np.array([data['cost'] for data in length_pair_metadata])
        rescaled_costs = self._rescale_costs(costs)
        
        utilities = []
        for i, length_meta in enumerate(length_pair_metadata):
            utility = self._calculate_single_audio_utility(
                length_meta, rescaled_costs[i], task
            )
            utilities.append(utility)
        
        probabilities = self._utilities_to_probabilities(np.array(utilities), task)
        return length_pair_metadata, probabilities
    
    def _calculate_single_audio_utility(
        self,
        length_meta: Dict,
        rescaled_cost: float,
        task: PathSamplingTask
    ) -> float:
        """Calculate utility for a single audio length pair"""
        
        if task.agent_type in ['naive', 'uniform']:
            return task.config.sampling.cost_weight * (1 - rescaled_cost)
        
        elif task.agent_type == 'sophisticated':
            eval_steps_to = length_meta['eval_steps_to']
            eval_steps_from = length_meta['eval_steps_from']
            
            # Use caching for efficiency
            cache_key = (eval_steps_to, eval_steps_from)
            if cache_key in self.audio_framing_cache:
                L_A_total, L_B_total = self.audio_framing_cache[cache_key]
            else:
                L_A_total, L_B_total = self._compute_audio_framing_likelihoods(
                    eval_steps_to, eval_steps_from, task.config
                )
                self.audio_framing_cache[cache_key] = (L_A_total, L_B_total)
            
            predicted_slider = normalized_slider_prediction(L_A_total, L_B_total)
            path_framing_metric_scaled = predicted_slider / 100.0
            cost_utility_comp = task.config.sampling.cost_weight * (1 - rescaled_cost)
            framing_utility_comp = (1 - task.config.sampling.cost_weight) * path_framing_metric_scaled
            
            if task.agent_id == 'A':
                return cost_utility_comp + framing_utility_comp
            else:
                return cost_utility_comp - framing_utility_comp
        
        return 0.0
    
    def _compute_audio_framing_likelihoods(
        self,
        eval_steps_to: int,
        eval_steps_from: int,
        config: SimulationConfig
    ) -> Tuple[float, float]:
        """Compute audio framing likelihoods for sophisticated agents"""
        
        sigma_factor = config.evidence.audio_similarity_sigma
        
        # Get model steps from config
        evidence_config = config.evidence
        model_A_to_steps = getattr(evidence_config, 'naive_A_to_fridge_steps_model', [])
        model_A_from_steps = getattr(evidence_config, 'naive_A_from_fridge_steps_model', [])
        model_B_to_steps = getattr(evidence_config, 'naive_B_to_fridge_steps_model', [])
        model_B_from_steps = getattr(evidence_config, 'naive_B_from_fridge_steps_model', [])
        
        # Calculate likelihoods
        L_A_to = np.mean([
            single_segment_audio_likelihood(model_s_to, eval_steps_to, sigma_factor)
            for model_s_to in model_A_to_steps
        ]) if model_A_to_steps else 0
        
        L_A_from = np.mean([
            single_segment_audio_likelihood(model_s_from, eval_steps_from, sigma_factor)
            for model_s_from in model_A_from_steps
        ]) if model_A_from_steps else 0
        
        L_B_to = np.mean([
            single_segment_audio_likelihood(model_s_to, eval_steps_to, sigma_factor)
            for model_s_to in model_B_to_steps
        ]) if model_B_to_steps else 0
        
        L_B_from = np.mean([
            single_segment_audio_likelihood(model_s_from, eval_steps_from, sigma_factor)
            for model_s_from in model_B_from_steps
        ]) if model_B_from_steps else 0
        
        L_A_total = L_A_to * L_A_from
        L_B_total = L_B_to * L_B_from
        
        return L_A_total, L_B_total
    
    def _group_paths_by_length(
        self,
        candidate_paths_to_fridge: List,
        candidate_paths_from_fridge: List
    ) -> Tuple[Dict[int, List], Dict[int, List]]:
        """Group paths by their lengths for efficient sampling"""
        
        paths_by_len_to = {}
        for p_seq in candidate_paths_to_fridge:
            p_len = max(0, len(p_seq) - 1 if p_seq else 0)
            if p_len not in paths_by_len_to:
                paths_by_len_to[p_len] = []
            paths_by_len_to[p_len].append(p_seq)
        
        paths_by_len_from = {}
        for p_seq in candidate_paths_from_fridge:
            p_len = max(0, len(p_seq) - 1 if p_seq else 0)
            if p_len not in paths_by_len_from:
                paths_by_len_from[p_len] = []
            paths_by_len_from[p_len].append(p_seq)
        
        return paths_by_len_to, paths_by_len_from
    
    def _sample_paths_with_lengths(
        self,
        paths_by_len_to: Dict[int, List],
        paths_by_len_from: Dict[int, List],
        selected_len_to: int,
        selected_len_from: int
    ) -> Tuple[Optional[List], Optional[List]]:
        """Sample paths with specific lengths"""
        
        possible_paths_to = paths_by_len_to.get(selected_len_to)
        possible_paths_from = paths_by_len_from.get(selected_len_from)
        
        if not possible_paths_to or not possible_paths_from:
            return None, None
        
        p_to_idx = np.random.randint(0, len(possible_paths_to))
        p_from_idx = np.random.randint(0, len(possible_paths_from))
        
        return possible_paths_to[p_to_idx], possible_paths_from[p_from_idx]
    
    def _utilities_to_probabilities(
        self,
        utilities: np.ndarray,
        task: PathSamplingTask
    ) -> np.ndarray:
        """Convert utilities to sampling probabilities"""
        
        if len(utilities) == 0:
            return np.array([])
        
        if np.all(utilities == utilities[0]):
            # All utilities are the same, use uniform distribution
            return np.full(len(utilities), 1.0 / len(utilities))
        
        temp = task.config.sampling.naive_temp if task.agent_type == 'naive' else task.config.sampling.sophisticated_temp
        return softmax_list_vals(utilities, temp=temp)
    
    def _rescale_costs(self, costs: np.ndarray) -> np.ndarray:
        """Rescale costs to [0, 1] range"""
        if len(costs) == 0:
            return costs
        
        min_cost, max_cost = np.min(costs), np.max(costs)
        if max_cost > min_cost:
            return (costs - min_cost) / (max_cost - min_cost)
        return np.zeros_like(costs, dtype=float)
    
    def _calculate_optimal_plant_spot_and_slider(
        self,
        world,
        agent_id: str,
        p2_seq: List,
        fridge_access_point,
        config: SimulationConfig
    ):
        """Calculate optimal plant spot for sophisticated visual agents"""
        optimal_plant_spot = None
        best_slider = -50.0 if agent_id == 'A' else 50.0
        valid_planting_spots = []

        if fridge_access_point and fridge_access_point in p2_seq:
            try:
                fridge_idx = p2_seq.index(fridge_access_point)
                for coord in p2_seq[fridge_idx:]:
                    if coord in world.world_graph.node_to_vid:
                        vid = world.world_graph.node_to_vid[coord]
                        node_attrs = world.world_graph.igraph.vs[vid]
                        if node_attrs['room'] == 'Kitchen' and not node_attrs['is_door']:
                            valid_planting_spots.append(coord)
            except ValueError:
                self.logger.warning(f"Fridge access point {fridge_access_point} not in p2_seq")
                return None, best_slider
        
        if not valid_planting_spots:
            return None, best_slider

        # Get naive maps from config
        naive_A_map = config.evidence.naive_A_visual_likelihoods_map
        naive_B_map = config.evidence.naive_B_visual_likelihoods_map
        
        if not naive_A_map or not naive_B_map:
            self.logger.warning("Naive likelihood maps not available for sophisticated agent")
            return None, best_slider

        # Find optimal spot based on agent strategy with improved tiebreaking
        other_agent_start = world.start_coords.get('B' if agent_id == 'A' else 'A')
        
        # Collect all optimal tiles instead of updating immediately
        optimal_tiles = []
        for tile in valid_planting_spots:
            l_A = naive_A_map.get(tuple(tile), 0.0)
            l_B = naive_B_map.get(tuple(tile), 0.0)
            slider = normalized_slider_prediction(l_A, l_B)

            is_better = (agent_id == 'A' and slider > best_slider) or (agent_id == 'B' and slider < best_slider)
            
            if is_better:
                # Found better slider value - reset collection
                best_slider = slider
                optimal_tiles = [tile]
            elif slider == best_slider:
                # Tied for best slider value
                optimal_tiles.append(tile)
        
        # Apply distance tiebreaker among optimal tiles
        if len(optimal_tiles) > 1 and other_agent_start is not None:
            min_distance = float('inf')
            closest_tiles = []
            
            for tile in optimal_tiles:
                distance = np.linalg.norm(np.array(tile) - np.array(other_agent_start))
                if distance < min_distance:
                    min_distance = distance
                    closest_tiles = [tile]
                elif distance == min_distance:
                    closest_tiles.append(tile)
            
            final_tiles = closest_tiles
        else:
            final_tiles = optimal_tiles
        
        # Random selection from final tied set
        if len(final_tiles) > 1:
            logging.info(f"Several 'closest optimal tiles' found: {final_tiles}")
            breakpoint()
            
        if final_tiles:
            optimal_plant_spot = final_tiles[np.random.randint(len(final_tiles))]
        
        return optimal_plant_spot, best_slider
    
    def _get_noisy_plant_spot(self, world, optimal_spot, sigma):
        """Get noisy plant spot for sophisticated agents"""
        potential_spots = [optimal_spot]
        ox, oy = optimal_spot
        
        # Check cardinal neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (ox + dx, oy + dy)
            if neighbor in world.world_graph.node_to_vid:
                vid = world.world_graph.node_to_vid[neighbor]
                node_attrs = world.world_graph.igraph.vs[vid]
                if node_attrs['room'] == 'Kitchen' and not node_attrs['is_door']:
                    potential_spots.append(neighbor)
        
        if len(potential_spots) == 1:
            return potential_spots[0]

        # Gaussian weighting
        weights = [np.exp(-((spot[0] - ox)**2 + (spot[1] - oy)**2) / (2 * sigma**2)) 
                  for spot in potential_spots]
        probabilities = np.array(weights) / np.sum(weights)
        chosen_idx = np.random.choice(len(potential_spots), p=probabilities)
        return potential_spots[chosen_idx]
    
    def _is_valid_audio_sequence(self, audio_seq) -> bool:
        """Validate audio sequence format"""
        return (isinstance(audio_seq, list) and 
                len(audio_seq) == 5 and 
                isinstance(audio_seq[0], int) and 
                isinstance(audio_seq[4], int))
    
    def _add_empty_audio_sample(self, result: SamplingResult):
        """Add empty entries for failed audio sampling"""
        result.audio_sequences.append(None)
        result.full_sequences.append([])
        result.to_fridge_sequences.append([])
        result.middle_sequences.append([])
        result.full_sequence_lengths.append(0)
        result.to_fridge_sequence_lengths.append(0)
        result.middle_sequence_lengths.append(0) 
        