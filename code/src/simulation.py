import logging
import pandas as pd
import ast
from utils import load_simple_path_sequences, save_sampled_paths_to_csv, smooth_likelihood_grid
from world import World
from agents import Suspect, Detective
from params import SimulationParams
import traceback


class BaseSimulator:
    def __init__(self, args, log_dir_base, param_log_dir, params: SimulationParams, trials_to_run):
        self.args = args
        self.log_dir_base = log_dir_base
        self.param_log_dir = param_log_dir
        self.params = params
        self.trials_to_run = trials_to_run
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_trial(self, trial_file: str, trial_name: str, w_t0: World, params: SimulationParams) -> dict:
        raise NotImplementedError("Subclasses must implement `run_trial()`")

    def run(self) -> list:
        self.logger.info(f"Starting simulation for {self.__class__.__name__}...")
        all_results = []

        for trial_file in self.trials_to_run:
            trial_name = trial_file.split('_A1.json')[0]
            self.logger.info(f"===== Running Trial: {trial_name} =====")
            try:
                w_t0 = World.initialize_world_start(trial_file)
                trial_result = self.run_trial(trial_file, trial_name, w_t0, self.params)
                if trial_result:
                    all_results.append(trial_result)
                self.logger.info(f"===== Finished Trial: {trial_name} =====")
            except Exception as e:
                self.logger.error(f"Error processing trial {trial_name}: {e}")
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Simulation for {self.__class__.__name__} completed.")
        return all_results


class RSMSimulator(BaseSimulator):
    def __init__(self, args, log_dir_base, param_log_dir, params: SimulationParams, trials_to_run):
        super().__init__(args, log_dir_base, param_log_dir, params, trials_to_run)

    def run_trial(self, trial_file: str, trial_name: str, w_t0: World, params: SimulationParams) -> dict:
        num_sample_paths_suspect = params.sample_paths_suspect
        num_sample_paths_for_detective = params.sample_paths_detective

        self.logger.info(f"Simulating {num_sample_paths_suspect} suspect paths per agent type.")
        self.logger.info(f"Simulating {num_sample_paths_for_detective} suspect paths for detective calculation.")

        simple_paths_A_seqs, simple_paths_B_seqs = load_simple_path_sequences(
            self.log_dir_base, trial_name, w_t0, params.max_steps
        )
        if simple_paths_A_seqs is None or simple_paths_B_seqs is None:
            self.logger.error(f"Failed to load or compute simple paths for {trial_name}. Skipping.")
            return None

        current_suspect_agent = Suspect(id='suspect_rsm', data_type='visual', params=params)
        current_detective_agent = Detective(id='detective_rsm', data_type='visual', params=params)

        ## Naive
        self.logger.info("--- Simulating Level 1 (Naive Agent) ---")
        # Suspect
        self.logger.info(f"Simulating {num_sample_paths_suspect} paths for suspect (naive)...")
        sampled_data_naive = current_suspect_agent.simulate_suspect(
            w_t0, simple_paths_A_seqs, simple_paths_B_seqs, 'naive',
            num_sample_paths_suspect, params
        )
        save_sampled_paths_to_csv(sampled_data_naive, trial_name, self.param_log_dir, 'naive')

        # Detective
        self.logger.info(f"Simulating {num_sample_paths_for_detective} paths for detective (modeling suspect as naive)...")
        sampled_data_for_naive_detective_calc = current_suspect_agent.simulate_suspect(
            w_t0, simple_paths_A_seqs, simple_paths_B_seqs, 'naive',
            num_sample_paths_for_detective, params
        )

        naive_slider_predictions_dict, naive_A_crumb_likelihoods_map_raw, naive_B_crumb_likelihoods_map_raw = current_detective_agent.simulate_detective(
            w_t0, trial_name, sampled_data_for_naive_detective_calc, 'naive', params, self.param_log_dir
        )

        if not naive_slider_predictions_dict:
            self.logger.warning(f"Skipping sophisticated agent for {trial_name} due to empty naive predictions.")
            return None

        suspect_sigma = params.soph_suspect_sigma
        naive_A_map_for_suspect = naive_A_crumb_likelihoods_map_raw
        naive_B_map_for_suspect = naive_B_crumb_likelihoods_map_raw

        if suspect_sigma > 0:
            self.logger.info(f"Smoothing naive likelihood maps for sophisticated suspect with sigma: {suspect_sigma}")
            naive_A_map_for_suspect = smooth_likelihood_grid(naive_A_crumb_likelihoods_map_raw, w_t0, suspect_sigma)
            naive_B_map_for_suspect = smooth_likelihood_grid(naive_B_crumb_likelihoods_map_raw, w_t0, suspect_sigma)

        ## Sophisticated
        self.logger.info("--- Simulating Level 2 (Sophisticated) ---")
        # Suspect
        self.logger.info(f"Simulating {num_sample_paths_suspect} paths for suspect (sophisticated)...")
        
        params.naive_A_crumb_likelihoods_map = naive_A_map_for_suspect
        params.naive_B_crumb_likelihoods_map = naive_B_map_for_suspect

        sampled_data_soph = current_suspect_agent.simulate_suspect(
            w_t0, simple_paths_A_seqs, simple_paths_B_seqs, 'sophisticated',
            num_sample_paths_suspect, params
        )
        save_sampled_paths_to_csv(sampled_data_soph, trial_name, self.param_log_dir, 'sophisticated')

        # Detective
        self.logger.info(f"Simulating {num_sample_paths_for_detective} paths for detective (modeling suspect as sophisticated)...")
        if num_sample_paths_for_detective != num_sample_paths_suspect:
            sampled_data_for_soph_detective_calc = current_suspect_agent.simulate_suspect(
                w_t0, simple_paths_A_seqs, simple_paths_B_seqs, 'sophisticated',
                num_sample_paths_for_detective, params
            )
        else:
            sampled_data_for_soph_detective_calc = sampled_data_soph

        soph_slider_predictions_dict, _, _ = current_detective_agent.simulate_detective(
            w_t0, trial_name, sampled_data_for_soph_detective_calc,
            'sophisticated', params, self.param_log_dir
        )

        return {
            "trial": trial_name,
            "naive_predictions": naive_slider_predictions_dict,
            "sophisticated_predictions": soph_slider_predictions_dict
        }


class EmpiricalSimulator(BaseSimulator):
    def __init__(self, args, log_dir_base, param_log_dir, params: SimulationParams, trials_to_run):
        super().__init__(args, log_dir_base, param_log_dir, params, trials_to_run)
        self.logger.info(f"Loading empirical paths from: {self.args.paths}")
        self.logger.info(f"Mismatched analysis: {self.args.mismatched}")
        self.all_empirical_paths_df = self._load_empirical_data(self.args.paths)

    def _load_empirical_data(self, paths_csv: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(paths_csv)
            required_cols = ['trial', 'agent', 'agent_type', 'full_sequence_world_coords', 'middle_sequence_world_coords']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Empirical CSV missing required columns: {missing_cols}")
                exit()

            full_path_col = 'full_sequence_world_coords'
            middle_path_col = 'middle_sequence_world_coords'
            df[full_path_col] = df[full_path_col].apply(ast.literal_eval)
            df[middle_path_col] = df[middle_path_col].apply(ast.literal_eval)
            self.logger.info(f"Successfully loaded and parsed {len(df)} empirical paths.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Empirical path file not found at: {paths_csv}")
            exit()
        except Exception as e:
            self.logger.error(f"Error loading empirical data: {e}")
            exit()

    def run_trial(self, trial_file: str, trial_name: str, w_t0: World, params: SimulationParams) -> dict:
        if self.all_empirical_paths_df is None:
            self.logger.error("Empirical data not loaded. Skipping trial.")
            return None

        trial_df_all = self.all_empirical_paths_df[self.all_empirical_paths_df['trial'] == trial_name]

        if trial_df_all.empty:
            self.logger.warning(f"No empirical paths found for trial {trial_name} in the CSV. Skipping.")
            return None

        detective_agent = Detective(id='detective_empirical', data_type='visual', params=params)

        path_data_by_type = {
            'naive': {'A': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}, 'B': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}},
            'sophisticated': {'A': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}, 'B': {'full_sequences': [], 'middle_sequences': [], 'chosen_plant_spots': []}}
        }
        has_data = {'naive': False, 'sophisticated': False}

        for agent_type_in_data in ['naive', 'sophisticated']:
            type_df = trial_df_all[trial_df_all['agent_type'] == agent_type_in_data]
            if not type_df.empty:
                has_data[agent_type_in_data] = True
                for _, row in type_df.iterrows():
                    agent = row['agent']
                    full_sequence = row['full_sequence_world_coords']
                    middle_sequence = row['middle_sequence_world_coords']
                    if agent in path_data_by_type[agent_type_in_data]:
                        path_data_by_type[agent_type_in_data][agent]['full_sequences'].append(full_sequence)
                        path_data_by_type[agent_type_in_data][agent]['middle_sequences'].append(middle_sequence)
                        path_data_by_type[agent_type_in_data][agent]['chosen_plant_spots'].append(None) 
                    else:
                        self.logger.warning(f"Unknown agent '{agent}' found in {agent_type_in_data} data for trial {trial_name}. Skipping row.")

        trial_results_payload = {"predictions": {}}
        for prediction_type in ['naive', 'sophisticated']:
            source_data_type = 'sophisticated' if self.args.mismatched and prediction_type == 'naive' else \
                             'naive' if self.args.mismatched and prediction_type == 'sophisticated' else \
                             prediction_type

            self.logger.info(f"Calculating {prediction_type.capitalize()} predictions using {source_data_type.capitalize()} empirical paths ---")

            if not has_data[source_data_type]:
                self.logger.warning(f"Source data type '{source_data_type}' needed for {prediction_type} prediction is missing for trial {trial_name}. Skipping.")
                trial_results_payload["predictions"][prediction_type] = {}
                continue

            data_for_detective = path_data_by_type[source_data_type]
            self.logger.info(f"Using paths: Agent A (Full: {len(data_for_detective['A']['full_sequences'])}) "
                               f"Agent B (Full: {len(data_for_detective['B']['full_sequences'])}) from {source_data_type} data.")

            predictions_dict_empirical, _, _ = detective_agent.simulate_detective(
                w_t0, trial_name, data_for_detective,
                agent_type=prediction_type,
                params=params,
                param_log_dir=self.param_log_dir,
                source_data_type=source_data_type,
                mismatched_run=self.args.mismatched
            )
            trial_results_payload["predictions"][prediction_type] = predictions_dict_empirical
        
        return {"trial": trial_name, **trial_results_payload}


class UniformSimulator(BaseSimulator):
    def __init__(self, args, log_dir_base, param_log_dir, params: SimulationParams, trials_to_run):
        super().__init__(args, log_dir_base, param_log_dir, params, trials_to_run)

    def run_trial(self, trial_file: str, trial_name: str, w_t0: World, params: SimulationParams) -> dict:
        num_uniform_samples_suspect = params.sample_paths_suspect
        num_uniform_samples_detective = params.sample_paths_detective

        suspect_agent = Suspect(id='suspect_uniform', data_type='visual', params=params)
        detective_agent = Detective(id='detective_uniform', data_type='visual', params=params)

        simple_paths_A_seqs, simple_paths_B_seqs = load_simple_path_sequences(self.log_dir_base, trial_name, w_t0, params.max_steps)

        if simple_paths_A_seqs is None or simple_paths_B_seqs is None:
            self.logger.error(f"Simple path sequences not found or failed to load for {trial_name}. Skipping uniform trial.")
            return None

        self.logger.info(f"Simulating {num_uniform_samples_suspect} paths for suspect (uniform)...")
        sampled_data_uniform_save = suspect_agent.simulate_suspect(
            w_t0, simple_paths_A_seqs, simple_paths_B_seqs, 'uniform',
            num_uniform_samples_suspect, params
        )
        save_sampled_paths_to_csv(sampled_data_uniform_save, trial_name, self.param_log_dir, 'uniform')

        self.logger.info(f"Simulating {num_uniform_samples_detective} paths for detective (uniform)...")
        sampled_data_uniform_detective_raw = suspect_agent.simulate_suspect(
            w_t0, simple_paths_A_seqs, simple_paths_B_seqs, 'uniform',
            num_uniform_samples_detective, params
        )
        
        sampled_data_uniform_detective = {}
        for agent_id_uniform, data_uniform in sampled_data_uniform_detective_raw.items():
            if 'full_sequences' in data_uniform:
                num_seqs_uniform = len(data_uniform['full_sequences'])
                data_uniform['chosen_plant_spots'] = [None] * num_seqs_uniform
            else:
                data_uniform['chosen_plant_spots'] = []
            sampled_data_uniform_detective[agent_id_uniform] = data_uniform

        uniform_predictions_dict, _, _ = detective_agent.simulate_detective(
            w_t0, trial_name, sampled_data_uniform_detective,
            'uniform', params, self.param_log_dir
        )

        return {
            "trial": trial_name,
            "uniform_predictions": uniform_predictions_dict
        }
