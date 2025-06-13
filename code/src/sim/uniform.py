import logging
import numpy as np
from src.core.world import World
from src.core.world.utils import load_or_compute_simple_path_sequences
from src.utils.io_utils import save_sampled_paths_to_csv
from src.core.evidence import get_compressed_audio_from_path
from src.agents import Suspect, Detective
from .base import BaseSimulator


class UniformSimulator(BaseSimulator):
    """
    Simulator for uniform recursive model.
    """
    def run_trial(self, trial_file: str, trial_name: str, world: World) -> dict:
        """Run uniform simulation for a single trial"""
        self.logger.info(f"Running uniform simulation for {trial_name}")
        
        # Get sampling parameters
        num_suspect_paths = self.config.sampling.num_suspect_paths
        num_detective_paths = self.config.sampling.num_detective_paths
        
        self.logger.info(f"Generating {num_suspect_paths} uniform suspect paths and {num_detective_paths} uniform detective paths")
        
        # Load path sequences for both agents
        paths_A, paths_B = load_or_compute_simple_path_sequences(world, trial_name, self.config, self.config.sampling.max_steps)
        
        # Initialize agents
        suspect = Suspect('suspect_uniform', self.config)
        detective = Detective('detective_uniform', self.config)
        
        # Sample uniform suspect paths
        self.logger.info("--- Simulating Uniform Suspects ---")
        uniform_suspect_data = {
            'A': self._sample_uniform_paths(world, paths_A, 'A', trial_name),
            'B': self._sample_uniform_paths(world, paths_B, 'B', trial_name)
        }
        save_sampled_paths_to_csv(uniform_suspect_data, trial_name, self.param_log_dir, 'uniform')
        
        # Sample uniform detective paths (for detective reasoning)
        self.logger.info("--- Simulating Uniform Detective ---")
        uniform_detective_data = {
            'A': self._sample_uniform_paths(world, paths_A, 'A', trial_name, num_detective_paths),
            'B': self._sample_uniform_paths(world, paths_B, 'B', trial_name, num_detective_paths)
        }
        
        # Detective reasoning: compute predictions based on uniform suspect behavior
        uniform_detective_result = detective.simulate_detective(
            world, uniform_detective_data, 'uniform', trial_name, self.param_log_dir
        )
        uniform_predictions = uniform_detective_result['predictions']
        
        return {
            "trial": trial_name,
            f"uniform_{self.config.evidence.evidence_type}_predictions": uniform_predictions
        }
    
    # TODO: move to correct file (maybe paths/sampling.py)
    def _sample_uniform_paths(self, world, path_sequences, agent_id: str, trial_name: str, num_paths: int = None) -> dict:
        """Sample paths uniformly for a single agent"""
        if num_paths is None:
            num_paths = self.config.sampling.num_suspect_paths
            
        p_start_door, p_door_fridge, p_fridge_door, p_door_start = path_sequences
        
        self.logger.info(f"Sampling {num_paths} uniform paths for agent {agent_id}")
        
        # Create all possible complete paths (start->door->fridge->door->start)
        all_to_fridge = [p1[:-1] + p2 for p1 in p_start_door for p2 in p_door_fridge]
        all_from_fridge = [p3[:-1] + p4 for p3 in p_fridge_door for p4 in p_door_start]
        
        # Sample results storage
        full_sequences = []
        to_fridge_sequences = []
        return_sequences = []
        middle_sequences = []
        audio_sequences = []
        chosen_plant_spots = []
        full_sequence_lengths = []
        to_fridge_sequence_lengths = []
        return_sequence_lengths = []
        middle_sequence_lengths = []
        
        for _ in range(num_paths):
            # Sample uniformly from all possible paths
            to_fridge_idx = np.random.randint(0, len(all_to_fridge))
            from_fridge_idx = np.random.randint(0, len(all_from_fridge))
            
            to_fridge_sequence = all_to_fridge[to_fridge_idx]
            return_sequence = all_from_fridge[from_fridge_idx]
            
            # Construct full path
            full_sequence = to_fridge_sequence[:-1] + return_sequence
            
            # Find the middle segment (fridge to door part)
            seg_fridge_door = next((p for p in p_fridge_door if tuple(return_sequence[-len(p):]) == tuple(p)), [])
            
            # Get audio representation  
            compressed_audio = get_compressed_audio_from_path(world, full_sequence)
            
            # Store results
            full_sequences.append(full_sequence)
            to_fridge_sequences.append(to_fridge_sequence)
            return_sequences.append(return_sequence)
            middle_sequences.append(seg_fridge_door)
            audio_sequences.append(compressed_audio)
            chosen_plant_spots.append(None)  # No plant spots for uniform
            
            full_sequence_lengths.append(len(full_sequence) - 1)
            to_fridge_sequence_lengths.append(len(to_fridge_sequence) - 1)
            return_sequence_lengths.append(len(return_sequence) - 1)
            middle_sequence_lengths.append(len(seg_fridge_door) - 1 if seg_fridge_door else 0)
        
        return {
            'full_sequences': full_sequences,
            'to_fridge_sequences': to_fridge_sequences,
            'return_sequences': return_sequences,
            'middle_sequences': middle_sequences,
            'audio_sequences': audio_sequences,
            'chosen_plant_spots': chosen_plant_spots,
            'full_sequence_lengths': full_sequence_lengths,
            'to_fridge_sequence_lengths': to_fridge_sequence_lengths,
            'return_sequence_lengths': return_sequence_lengths,
            'middle_sequence_lengths': middle_sequence_lengths
        }
    