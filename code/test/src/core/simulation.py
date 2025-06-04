import logging
import pandas as pd
import ast
from src.utils.math_utils import smooth_likelihood_grid, smooth_likelihood_grid_connectivity_aware
from src.utils.io_utils import save_sampled_paths_to_csv
from src.utils.world_utils import PathSequenceCache
from src.core.world import World, load_simple_path_sequences
from src.core.agents import Suspect, Detective
from src.configs import SimulationConfig
import traceback


class BaseSimulator:
    """Base class for all simulation types"""
    
    def __init__(self, args, log_dir_base, param_log_dir, cfg: SimulationConfig, trials_to_run):
        self.args = args
        self.log_dir_base = log_dir_base
        self.param_log_dir = param_log_dir
        self.cfg = cfg
        self.trials_to_run = trials_to_run
        self.logger = logging.getLogger(self.__class__.__name__)
        self.path_cache = PathSequenceCache(log_dir_base)

    def run_trial(self, trial_file: str, trial_name: str, world: World) -> dict:
        """Run simulation for a single trial. Must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement `run_trial()`")

    def run(self) -> list:
        """Run simulation for all trials"""
        self.logger.info(f"Starting {self.__class__.__name__} simulation with {self.cfg.evidence.type} evidence type")
        all_results = []

        for trial_name in self.trials_to_run:
            self.logger.info(f"===== Running Trial: {trial_name} =====")
            
            try:
                # Update config with current trial name
                trial_config = self.cfg
                trial_config.trial.name = trial_name
                
                # Initialize world using config object for proper file searching
                world = World.initialize_world_start(trial_config)
                trial_result = self.run_trial(trial_name, trial_name, world)
                if trial_result:
                    all_results.append(trial_result)
                self.logger.info(f"===== Finished Trial: {trial_name} =====")
            except Exception as e:
                self.logger.error(f"Error processing trial {trial_name}: {e}")
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"{self.__class__.__name__} simulation completed")
        return all_results


class RSMSimulator(BaseSimulator):
    """Rational Speech-act Model simulator for suspects and detectives"""
    
    def run_trial(self, trial_name: str, trial_display_name: str, world: World) -> dict:
        """Run RSM simulation for a single trial"""
        num_suspect_paths = self.cfg.sampling.num_suspect_paths
        num_detective_paths = self.cfg.sampling.num_detective_paths

        self.logger.info(f"Generating {num_suspect_paths} suspect paths and {num_detective_paths} detective paths")

        # Load path sequences for both agents
        paths_A, paths_B = load_simple_path_sequences(self.log_dir_base, trial_name, world, self.cfg, self.cfg.sampling.max_steps)

        # Initialize agents
        suspect = Suspect('suspect_rsm', self.cfg.evidence.type, self.cfg)
        detective = Detective('detective_rsm', self.cfg.evidence.type, self.cfg)

        # Level 1: Naive agents
        self.logger.info("--- Simulating Level 1 (Naive Agent) ---")
        
        naive_suspect_data = suspect.simulate_suspect(world, paths_A, paths_B, 'naive', num_suspect_paths)
        save_sampled_paths_to_csv(naive_suspect_data, trial_name, self.param_log_dir, 'naive')

        naive_detective_data = suspect.simulate_suspect(world, paths_A, paths_B, 'naive', num_detective_paths)

        naive_predictions, naive_A_model, naive_B_model = detective.simulate_detective(world, trial_name, naive_detective_data, 'naive', self.param_log_dir)

        # Process naive models for sophisticated suspects
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

        soph_predictions, _, _ = detective.simulate_detective(world, trial_name, soph_detective_data, 'sophisticated', self.param_log_dir)

        return {
            "trial": trial_name,
            f"naive_{self.cfg.evidence.type}_predictions": naive_predictions,
            f"sophisticated_{self.cfg.evidence.type}_predictions": soph_predictions
        }

    def _process_naive_models_for_sophisticated(self, naive_A_model, naive_B_model, world):
        """Process naive detective models for use by sophisticated suspects"""
        if self.cfg.evidence.type == 'visual':
            # Apply smoothing if specified
            smoothed_A_map = naive_A_model
            smoothed_B_map = naive_B_model
            
            suspect_sigma = self.cfg.evidence.sophisticated_suspect_sigma
            if suspect_sigma > 0 and naive_A_model and naive_B_model:
                self.logger.info(f"Smoothing visual likelihood maps (sigma={suspect_sigma})")
                
                # Connectivity-aware smoothing
                sigma_steps = max(1, int(suspect_sigma))  # Convert sigma to discrete steps
                smoothed_A_map = smooth_likelihood_grid_connectivity_aware(naive_A_model, world, sigma_steps)
                smoothed_B_map = smooth_likelihood_grid_connectivity_aware(naive_B_model, world, sigma_steps)
            
            # Store in evidence config
            self.cfg.evidence.naive_A_visual_likelihoods_map = smoothed_A_map
            self.cfg.evidence.naive_B_visual_likelihoods_map = smoothed_B_map
            
        elif self.cfg.evidence.type == 'audio':
            # Store audio step models for sophisticated suspects
            # Handle both dummy empty data and real data
            if isinstance(naive_A_model, tuple) and len(naive_A_model) >= 2:
                self.cfg.evidence.naive_A_to_fridge_steps_model = naive_A_model[0]
                self.cfg.evidence.naive_A_from_fridge_steps_model = naive_A_model[1]
            else:
                self.cfg.evidence.naive_A_to_fridge_steps_model = []
                self.cfg.evidence.naive_A_from_fridge_steps_model = []
                
            if isinstance(naive_B_model, tuple) and len(naive_B_model) >= 2:
                self.cfg.evidence.naive_B_to_fridge_steps_model = naive_B_model[0]
                self.cfg.evidence.naive_B_from_fridge_steps_model = naive_B_model[1]
            else:
                self.cfg.evidence.naive_B_to_fridge_steps_model = []
                self.cfg.evidence.naive_B_from_fridge_steps_model = []
            
            self.logger.info(f"Audio models: A_to({len(self.cfg.evidence.naive_A_to_fridge_steps_model)}), "
                           f"A_from({len(self.cfg.evidence.naive_A_from_fridge_steps_model)}), "
                           f"B_to({len(self.cfg.evidence.naive_B_to_fridge_steps_model)}), "
                           f"B_from({len(self.cfg.evidence.naive_B_from_fridge_steps_model)})")


class EmpiricalSimulator(BaseSimulator):
    """Simulator for empirical data analysis"""
    
    def __init__(self, args, log_dir_base, param_log_dir, cfg: SimulationConfig, trials_to_run):
        super().__init__(args, log_dir_base, param_log_dir, cfg, trials_to_run)
        self.logger.info(f"Loading empirical paths from: {self.cfg.empirical_paths}")
        self.logger.info(f"Mismatched analysis: {self.cfg.mismatched_analysis}")
        
        if not self.cfg.empirical_paths:
            self.logger.error("Empirical path CSV file not specified")
            self.empirical_data = pd.DataFrame()
        else:
            self.empirical_data = self._load_empirical_data(self.cfg.empirical_paths)

    def _load_empirical_data(self, paths_csv: str) -> pd.DataFrame:
        """Load and validate empirical path data"""
        try:
            df = pd.read_csv(paths_csv)
            required_cols = ['trial', 'agent', 'agent_type', 'full_sequence_world_coords', 'middle_sequence_world_coords']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                exit()

            # Parse coordinate columns
            df['full_sequence_world_coords'] = df['full_sequence_world_coords'].apply(ast.literal_eval)
            df['middle_sequence_world_coords'] = df['middle_sequence_world_coords'].apply(ast.literal_eval)
            
            self.logger.info(f"Loaded {len(df)} empirical paths")
            return df
            
        except FileNotFoundError:
            self.logger.error(f"Empirical file not found: {paths_csv}")
            exit()
        except Exception as e:
            self.logger.error(f"Error loading empirical data: {e}")
            exit()

    def run_trial(self, trial_file: str, trial_name: str, world: World) -> dict:
        """Run empirical analysis for a single trial"""
        trial_data = self.empirical_data[self.empirical_data['trial'] == trial_name]

        if trial_data.empty:
            self.logger.warning(f"No empirical data found for trial {trial_name}")
            return None

        detective = Detective('detective_empirical', 'visual', self.cfg)

        # Organize path data by agent type
        path_data = self._organize_path_data(trial_data)
        
        # Run predictions for both naive and sophisticated
        results = {"trial": trial_name, "predictions": {}}
        
        for prediction_type in ['naive', 'sophisticated']:
            source_type = self._get_source_type(prediction_type)
            
            if not self._has_data_for_type(path_data, source_type):
                self.logger.warning(f"No {source_type} data for {prediction_type} prediction in {trial_name}")
                results["predictions"][prediction_type] = {}
                continue

            self.logger.info(f"Computing {prediction_type} predictions using {source_type} data")
            
            predictions, _, _ = detective.simulate_detective(
                world, trial_name, path_data[source_type], prediction_type, self.param_log_dir,
                source_data_type=source_type, mismatched_run=self.args.mismatched
            )
            results["predictions"][prediction_type] = predictions
        
        return results

    def _organize_path_data(self, trial_data):
        """Organize trial data by agent type"""
        path_data = {
            'naive': {'A': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}, 
                     'B': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}},
            'sophisticated': {'A': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}, 
                            'B': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}}
        }

        for _, row in trial_data.iterrows():
            agent_type = row['agent_type']
            agent = row['agent']
            
            if agent_type in path_data and agent in path_data[agent_type]:
                path_data[agent_type][agent]['full_sequences'].append(row['full_sequence_world_coords'])
                path_data[agent_type][agent]['middle_sequences'].append(row['middle_sequence_world_coords'])
                path_data[agent_type][agent]['chosen_plant_spots'].append(None)

        return path_data

    def _get_source_type(self, prediction_type):
        """Determine source data type based on prediction type and mismatch setting"""
        if self.args.mismatched:
            return 'sophisticated' if prediction_type == 'naive' else 'naive'
        return prediction_type

    def _has_data_for_type(self, path_data, data_type):
        """Check if path data exists for given type"""
        return (data_type in path_data and 
                any(len(path_data[data_type][agent]['full_sequences']) > 0 
                    for agent in ['A', 'B']))


class UniformSimulator(BaseSimulator):
    """Uniform random simulator for baseline comparisons"""
    
    def run_trial(self, trial_file: str, trial_name: str, world: World) -> dict:
        """Run uniform simulation for a single trial"""
        num_suspect_paths = self.cfg.sampling.num_suspect_paths
        num_detective_paths = self.cfg.sampling.num_suspect_paths

        suspect = Suspect('suspect_uniform', 'visual', self.cfg)
        detective = Detective('detective_uniform', 'visual', self.cfg)

        # Load path sequences
        paths_A, paths_B = load_simple_path_sequences(self.log_dir_base, trial_name, world, self.cfg, self.cfg.sampling.max_steps)

        if not paths_A or not paths_B:
            self.logger.error(f"Path sequences not found for {trial_name}")
            return None

        # Generate uniform suspect paths
        self.logger.info(f"Generating {num_suspect_paths} uniform suspect paths")
        suspect_data = suspect.simulate_suspect(
            world, paths_A, paths_B, 'uniform', num_suspect_paths
        )
        save_sampled_paths_to_csv(suspect_data, trial_name, self.param_log_dir, 'uniform')

        # Generate uniform detective paths
        self.logger.info(f"Generating {num_detective_paths} uniform detective paths")
        detective_data = suspect.simulate_suspect(
            world, paths_A, paths_B, 'uniform', num_detective_paths
        )
        
        # Add plant spots for visual cfg.evidence.type compatibility
        for agent_id, data in detective_data.items():
            if 'full_sequences' in data:
                data['chosen_plant_spots'] = [None] * len(data['full_sequences'])
            else:
                data['chosen_plant_spots'] = []

        # Generate uniform predictions
        predictions, _, _ = detective.simulate_detective(
            world, trial_name, detective_data, 'uniform', self.param_log_dir
        )

        return {
            "trial": trial_name,
            "uniform_predictions": predictions
        }
