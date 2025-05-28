import logging
import pandas as pd
import ast
from test_utils import save_sampled_paths_to_csv, smooth_likelihood_grid
from test_world import World, load_simple_path_sequences
from test_agents import Suspect, Detective
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

    def run_trial(self, trial_file: str, trial_name: str, w_t0: World) -> dict:
        raise NotImplementedError("Subclasses must implement `run_trial()`")

    def run(self) -> list:
        self.logger.info(f"Starting simulation for {self.__class__.__name__} with evidence: {self.params.evidence}...")
        all_results = []

        for trial_file in self.trials_to_run:
            trial_name = trial_file.split('_A1.json')[0]
            self.logger.info(f"===== Running Trial: {trial_name} =====")
            try:
                w_t0 = World.initialize_world_start(trial_file)
                trial_result = self.run_trial(trial_file, trial_name, w_t0)
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

    def run_trial(self, trial_file: str, trial_name: str, w_t0: World) -> dict:
        num_sample_paths_suspect = self.params.sample_paths_suspect
        num_sample_paths_for_detective = self.params.sample_paths_detective

        self.logger.info(f"Simulating {num_sample_paths_suspect} suspect paths per agent type for {self.params.evidence} evidence.")
        self.logger.info(f"Simulating {num_sample_paths_for_detective} suspect paths for detective calculation.")

        # Load simple path sequences for both agents A and B for the current trial
        # load_simple_path_sequences now returns (paths_A_tuple, paths_B_tuple)
        # where each tuple is (p1_seqs, p2_seqs, p3_seqs, p_fs_seqs)
        paths_A_segments, paths_B_segments = load_simple_path_sequences(self.log_dir_base, trial_name, w_t0, self.params, self.params.max_steps)

        if not paths_A_segments or not paths_B_segments: # Basic check, detailed checks are in load_simple_path_sequences
            self.logger.error(f"Failed to load or compute simple path sequences for {trial_name}. Skipping.")
            return None
        
        # Unpack for clarity, though direct indexing like paths_A_segments[0] for P1_A is also possible
        simple_paths_A_p1, simple_paths_A_p2, simple_paths_A_p3, simple_paths_A_fs = paths_A_segments
        simple_paths_B_p1, simple_paths_B_p2, simple_paths_B_p3, simple_paths_B_fs = paths_B_segments

        # For audio evidence, the suspect simulation will now use simple_paths_A_p1 and simple_paths_A_fs (and similar for B)
        # The current Suspect.simulate_suspect takes a single 'simple_path_sequences' argument.
        # This will need to be adapted or the information packed differently if simulate_suspect needs all segments.
        # For the new audio logic in World.get_sample_paths, it expects a tuple of sequences:
        # (candidate_paths_to_fridge, candidate_paths_from_fridge_to_start)
        # So, we should pass these specific segments for audio.

        # For visual, it used (P1, P2, P3).
        # We need to decide how to pass these to simulate_suspect and then to get_sample_paths.
        # For now, let's assemble what each evidence type would primarily expect for its path choices.

        # This is what Suspect.simulate_suspect expects: a list/tuple of path segments
        # For visual, it was (p1, p2, p3)
        # For new audio, it will be (p1, p_fs)
        # We'll pass all 4 for now, and let get_sample_paths pick what it needs based on evidence type.
        # This avoids changing simulate_suspect signature immediately.
        # The World.get_sample_paths will then unpack this tuple of 4 lists.

        current_suspect_agent = Suspect(id='suspect_rsm', data_type=self.params.evidence, params=self.params)
        current_detective_agent = Detective(id='detective_rsm', data_type=self.params.evidence, params=self.params)

        ## Naive Simulation
        self.logger.info("--- Simulating Level 1 (Naive Agent) ---")
        # Naive Suspects (generates paths based on naive utility)
        self.logger.info(f"Simulating {num_sample_paths_suspect} paths for NAIVE suspect...")
        sampled_data_naive_suspect = current_suspect_agent.simulate_suspect(
            w_t0, 
            paths_A_segments, # Pass the tuple (p1,p2,p3,p_fs) for A
            paths_B_segments, # Pass the tuple (p1,p2,p3,p_fs) for B
            'naive',
            num_sample_paths_suspect
        )
        save_sampled_paths_to_csv(sampled_data_naive_suspect, trial_name, self.param_log_dir, 'naive')

        # Naive Detective
        self.logger.info(f"Simulating {num_sample_paths_for_detective} paths for NAIVE DETECTIVE's internal model...")
        sampled_data_for_naive_detective_model = current_suspect_agent.simulate_suspect(
            w_t0, 
            paths_A_segments, 
            paths_B_segments, 
            'naive',
            num_sample_paths_for_detective
        )

        naive_predictions_output, naive_A_model_for_soph_suspect, naive_B_model_for_soph_suspect = current_detective_agent.simulate_detective(
            w_t0, trial_name, sampled_data_for_naive_detective_model, 'naive', self.param_log_dir
        )

        if self.params.evidence == 'visual':
            smoothed_A_map = naive_A_model_for_soph_suspect
            smoothed_B_map = naive_B_model_for_soph_suspect
            if self.params.soph_suspect_sigma > 0:
                self.logger.info(f"Smoothing naive visual likelihood maps for sophisticated suspect with sigma: {self.params.soph_suspect_sigma}")
                smoothed_A_map = smooth_likelihood_grid(naive_A_model_for_soph_suspect, w_t0, self.params.soph_suspect_sigma)
                smoothed_B_map = smooth_likelihood_grid(naive_B_model_for_soph_suspect, w_t0, self.params.soph_suspect_sigma)
            self.params.naive_A_visual_likelihoods_map = smoothed_A_map
            self.params.naive_B_visual_likelihoods_map = smoothed_B_map
        elif self.params.evidence == 'audio':
            # naive_A_model_for_soph_suspect is now (list_of_to_steps, list_of_from_steps)
            # naive_B_model_for_soph_suspect is now (list_of_to_steps, list_of_from_steps)
            self.params.naive_A_to_fridge_steps_model = naive_A_model_for_soph_suspect[0]
            self.params.naive_A_from_fridge_steps_model = naive_A_model_for_soph_suspect[1]
            self.params.naive_B_to_fridge_steps_model = naive_B_model_for_soph_suspect[0]
            self.params.naive_B_from_fridge_steps_model = naive_B_model_for_soph_suspect[1]
            self.logger.info(f"Naive audio models for sophisticated suspect: "
                             f"A_to ({len(self.params.naive_A_to_fridge_steps_model)} steps), "
                             f"A_from ({len(self.params.naive_A_from_fridge_steps_model)} steps), "
                             f"B_to ({len(self.params.naive_B_to_fridge_steps_model)} steps), "
                             f"B_from ({len(self.params.naive_B_from_fridge_steps_model)} steps).")
            self.logger.info(f"Content of naive_predictions_output for trial {trial_name} (audio): {naive_predictions_output}")

        ## Sophisticated Simulation
        self.logger.info("--- Simulating Level 2 (Sophisticated) ---")
        # Sophisticated Suspect (generates paths considering the naive detective's model)
        self.logger.info(f"Simulating {num_sample_paths_suspect} paths for SOPHISTICATED suspect...")
        # Sophisticated suspect internally uses self.params.naive_X_visual_likelihoods_map or self.params.naive_X_audio_sequences
        sampled_data_soph_suspect = current_suspect_agent.simulate_suspect(
            w_t0, 
            paths_A_segments, 
            paths_B_segments, 
            'sophisticated',
            num_sample_paths_suspect
        )
        save_sampled_paths_to_csv(sampled_data_soph_suspect, trial_name, self.param_log_dir, 'sophisticated')

        # Sophisticated Detective (models sophisticated suspects)
        # This run is to get the final predictions about sophisticated suspects.
        self.logger.info(f"Simulating {num_sample_paths_for_detective} paths for SOPHISTICATED DETECTIVE's internal model...")
        # The paths generated here are by *sophisticated suspects* because the sophisticated detective models sophisticated suspects.
        # Re-sample if num_sample_paths_for_detective is different, or reuse if same.
        if num_sample_paths_for_detective != num_sample_paths_suspect:
            sampled_data_for_soph_detective_model = current_suspect_agent.simulate_suspect(
                w_t0, 
                paths_A_segments, 
                paths_B_segments, 
                'sophisticated',
                num_sample_paths_for_detective
            )
        else:
            sampled_data_for_soph_detective_model = sampled_data_soph_suspect

        soph_predictions_output, _, _ = current_detective_agent.simulate_detective( # We only need predictions here
            w_t0, trial_name, sampled_data_for_soph_detective_model,
            'sophisticated', self.param_log_dir
        )

        return {
            "trial": trial_name,
            f"naive_{self.params.evidence}_predictions": naive_predictions_output,
            f"sophisticated_{self.params.evidence}_predictions": soph_predictions_output
        }


class EmpiricalSimulator(BaseSimulator):
    def __init__(self, args, log_dir_base, param_log_dir, params: SimulationParams, trials_to_run):
        super().__init__(args, log_dir_base, param_log_dir, params, trials_to_run)
        self.logger.info(f"Loading empirical paths from: {self.params.paths}")
        self.logger.info(f"Mismatched analysis: {self.params.mismatched}")
        if not self.params.paths:
            self.logger.error("Empirical path CSV file not specified in params.")
            self.all_empirical_paths_df = pd.DataFrame()
        else:
            self.all_empirical_paths_df = self._load_empirical_data(self.params.paths)

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
