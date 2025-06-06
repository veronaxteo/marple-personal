from src.utils.math_utils import compute_all_graph_neighbors, smooth_likelihoods
from src.utils.io_utils import save_sampled_paths_to_csv
from src.core.world import World
from src.core.world.utils import load_or_compute_simple_path_sequences
from src.agents import Suspect, Detective
from .base import BaseSimulator


class RSMSimulator(BaseSimulator):
    """
    Recursive simulation model for up to level k=2 suspects and detectives.
    """
    def run_trial(self, trial_file: str, trial_name: str, world: World) -> dict:
        """Run RSM simulation for a single trial"""
        num_suspect_paths = self.config.sampling.num_suspect_paths
        num_detective_paths = self.config.sampling.num_detective_paths

        self.logger.info(f"Generating {num_suspect_paths} suspect paths and {num_detective_paths} detective paths")

        # Load path sequences for both agents
        paths_A, paths_B = load_or_compute_simple_path_sequences(
            world, trial_name, self.config, self.config.sampling.max_steps
        )

        # Initialize agents
        suspect = Suspect('suspect_rsm', self.config)
        detective = Detective('detective_rsm', self.config)

        # Level 1: Naive agents
        self.logger.info("--- Simulating Level 1 (Naive Agent) ---")
        
        naive_suspect_data = suspect.simulate_suspect(world, paths_A, paths_B, 'naive', num_suspect_paths)
        save_sampled_paths_to_csv(naive_suspect_data, trial_name, self.param_log_dir, 'naive')

        naive_detective_data = suspect.simulate_suspect(world, paths_A, paths_B, 'naive', num_detective_paths)

        naive_detective_result = detective.simulate_detective(world, naive_detective_data, 'naive', trial_name, self.param_log_dir)
        
        naive_predictions = naive_detective_result['predictions']
        naive_A_model = naive_detective_result['model_output_A'] 
        naive_B_model = naive_detective_result['model_output_B']

        # Process naive models for sophisticated suspects
        # Note: smoothing only done after naive detective predictions (smoothing for sophisticated agents only)
        self._process_naive_models_for_sophisticated(naive_A_model, naive_B_model, world)
        
        # Level 2: Sophisticated agents
        self.logger.info("--- Simulating Level 2 (Sophisticated Agent) ---")
        
        soph_suspect_data = suspect.simulate_suspect(world, paths_A, paths_B, 'sophisticated', num_suspect_paths)
        save_sampled_paths_to_csv(soph_suspect_data, trial_name, self.param_log_dir, 'sophisticated')

        # Generate sophisticated detective predictions
        if num_detective_paths != num_suspect_paths:
            soph_detective_data = suspect.simulate_suspect(world, paths_A, paths_B, 'sophisticated', num_detective_paths)
        else:
            soph_detective_data = soph_suspect_data

        soph_detective_result = detective.simulate_detective(world, soph_detective_data, 'sophisticated', trial_name, self.param_log_dir)
        soph_predictions = soph_detective_result['predictions']

        return {
            "trial": trial_name,
            f"naive_{self.config.evidence.evidence_type}_predictions": naive_predictions,
            f"sophisticated_{self.config.evidence.evidence_type}_predictions": soph_predictions
        }

    def _process_naive_models_for_sophisticated(self, naive_A_model, naive_B_model, world):
        """Process naive detective models for use by sophisticated suspects."""
        if self.config.evidence.evidence_type == 'visual':
            smoothed_A_map = naive_A_model
            smoothed_B_map = naive_B_model
            
            if self.config.evidence.naive_detective_sigma > 0:
                self.logger.info(f"Smoothing visual likelihood maps (sigma={self.config.evidence.naive_detective_sigma})")
                # Smoothing
                sigma_steps = max(1, int(self.config.evidence.naive_detective_sigma))
                neighbors = compute_all_graph_neighbors(world, naive_A_model.keys())
                smoothed_A_map = smooth_likelihoods(naive_A_model, sigma_steps, neighbors)
                smoothed_B_map = smooth_likelihoods(naive_B_model, sigma_steps, neighbors)
            
            self.config.evidence.naive_A_visual_likelihoods_map = smoothed_A_map
            self.config.evidence.naive_B_visual_likelihoods_map = smoothed_B_map
            
        elif self.config.evidence.evidence_type == 'audio':
            # Store audio step models for sophisticated suspects
            self.config.evidence.naive_A_to_fridge_steps_model = naive_A_model[0]
            self.config.evidence.naive_A_from_fridge_steps_model = naive_A_model[1]
            self.config.evidence.naive_B_to_fridge_steps_model = naive_B_model[0]
            self.config.evidence.naive_B_from_fridge_steps_model = naive_B_model[1]
            
            self.logger.info(f"Audio models: A_to({len(self.config.evidence.naive_A_to_fridge_steps_model)}), "
                           f"A_from({len(self.config.evidence.naive_A_from_fridge_steps_model)}), "
                           f"B_to({len(self.config.evidence.naive_B_to_fridge_steps_model)}), "
                           f"B_from({len(self.config.evidence.naive_B_from_fridge_steps_model)})")

        elif self.config.evidence.evidence_type == 'multimodal':            
            # Visual component
            naive_A_visual_model = naive_A_model['visual']
            naive_B_visual_model = naive_B_model['visual']
            
            smoothed_A_map = naive_A_visual_model
            smoothed_B_map = naive_B_visual_model
            
            if self.config.evidence.naive_detective_sigma > 0:
                self.logger.info(f"Smoothing visual likelihood maps (sigma={self.config.evidence.naive_detective_sigma})")
                sigma_steps = max(1, int(self.config.evidence.naive_detective_sigma))
                neighbors = compute_all_graph_neighbors(world, naive_A_visual_model.keys())
                smoothed_A_map = smooth_likelihoods(naive_A_visual_model, sigma_steps, neighbors)
                smoothed_B_map = smooth_likelihoods(naive_B_visual_model, sigma_steps, neighbors)
            
            self.config.evidence.naive_A_visual_likelihoods_map = smoothed_A_map
            self.config.evidence.naive_B_visual_likelihoods_map = smoothed_B_map
            
            # Audio component
            naive_A_audio_model = naive_A_model['audio']
            naive_B_audio_model = naive_B_model['audio']

            self.config.evidence.naive_A_to_fridge_steps_model = naive_A_audio_model[0]
            self.config.evidence.naive_A_from_fridge_steps_model = naive_A_audio_model[1]
            self.config.evidence.naive_B_to_fridge_steps_model = naive_B_audio_model[0]
            self.config.evidence.naive_B_from_fridge_steps_model = naive_B_audio_model[1]
            
            self.logger.info(f"Visual and Audio models loaded for multimodal simulation.")
