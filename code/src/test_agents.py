import logging
import pandas as pd
import numpy as np
import json
import os

from test_world import World
from params import SimulationParams
from test_evidence import VisualEvidence, AudioEvidence, get_compressed_audio_from_path, single_segment_audio_likelihood, generate_ground_truth_audio_sequences
from test_utils import normalized_slider_prediction, smooth_likelihood_grid, ensure_serializable, save_grid_to_json


class Agent:
    """
    The base agent class.

    Attributes:
        id (str): agent id (e.g., 'A', 'B')
        data_type (str): type of data to use for evidence (e.g., visual, audio, multimodal)
        params (dict): parameters for the agent, including agent type (e.g., naive, sophisticated), w, temp, etc.
    """
    def __init__(self, id, data_type, params: SimulationParams):
        self.id = id
        self.data_type = data_type
        self.params = params
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.id}]")


class Suspect(Agent):
    """
    Suspect agent class. 

    Attributes:
        id (inherited): str, agent id (e.g., 'A', 'B')
        data_type (inherited): str, type of data to use for evidence (e.g., visual, audio, multimodal)
        params (inherited): dict, parameters for the agent, including agent type (e.g., naive, sophisticated), w, temp, etc.
    """
    def __init__(self, id, data_type, params):
        super().__init__(id, data_type, params)
        self.world_state = None
        self.audio_evidence = None
        self.visual_evidence = None


    def simulate_suspect(
        self, 
        w_t0: World, 
        simple_paths_A_seqs: list, 
        simple_paths_B_seqs: list, 
        agent_type: str, # 'naive' or 'sophisticated' (or 'uniform')
        num_sample_paths: int, 
        # Params like naive_A_crumb_likelihoods_map, naive_B_crumb_likelihoods_map,
        # naive_A_audio_sequences, naive_B_audio_sequences are now accessed via self.params
        # w, temp, noisy_planting_sigma are also via self.params
    ) -> dict:
        self.logger.info(f"Simulating Suspect ({agent_type}) paths using {self.params.evidence} evidence.")
        
        agent_A_paths_data = w_t0.get_sample_paths(
            agent_id='A', 
            simple_path_sequences=simple_paths_A_seqs, 
            num_sample_paths=num_sample_paths, 
            params=self.params, # Pass the full params object
            agent_type=agent_type 
        )
        agent_B_paths_data = w_t0.get_sample_paths(
            agent_id='B', 
            simple_path_sequences=simple_paths_B_seqs, 
            num_sample_paths=num_sample_paths, 
            params=self.params, # Pass the full params object
            agent_type=agent_type
        )
        return {'A': agent_A_paths_data, 'B': agent_B_paths_data}


class Detective(Agent):
    """
    Detective agent class. 

    Attributes:
        id (inherited): str, agent id (e.g., 'A', 'B')
        data_type (inherited): str, type of data to use for evidence (e.g., visual, audio, multimodal)
        params (inherited): dict, parameters for the agent, including agent type (e.g., naive, sophisticated), w, temp, etc.
    """
    def __init__(self, id, data_type, params):
        super().__init__(id, data_type, params)
        self.world_state = None
        self.audio_evidence = None
        self.visual_evidence = None
    

    def compute_likelihoods(self, w_t0: World, sampled_data: dict, agent_type: str, params: SimulationParams, possible_crumb_coords: list) -> tuple[dict, dict]:
        raw_likelihoods_A = {}
        raw_likelihoods_B = {}
        for crumb_tuple in possible_crumb_coords:
            for agent_id in ['A', 'B']:
                agent_specific_data = sampled_data.get(agent_id, {})
                full_sequences = agent_specific_data.get('full_sequences', [])
                middle_sequences = agent_specific_data.get('middle_sequences', [])
                chosen_plant_spots = agent_specific_data.get('chosen_plant_spots', [])
                
                current_chosen_plant_spots = chosen_plant_spots if agent_type == 'sophisticated' else None

                likelihood_value = VisualEvidence.get_visual_evidence_likelihood(
                    crumb_coord_tuple=crumb_tuple,
                    agent_full_sequences=full_sequences,
                    agent_middle_sequences=middle_sequences,
                    world_state=w_t0,
                    agent_type_being_simulated=agent_type,
                    chosen_plant_spots_for_sequences=current_chosen_plant_spots
                )
                if agent_id == 'A':
                    raw_likelihoods_A[crumb_tuple] = likelihood_value
                else:
                    raw_likelihoods_B[crumb_tuple] = likelihood_value
        return raw_likelihoods_A, raw_likelihoods_B


    def smooth_likelihoods(self, raw_likelihoods_A: dict, raw_likelihoods_B: dict, agent_type: str, w_t0: World, params: SimulationParams) -> tuple[dict, dict]:
        if agent_type == 'sophisticated':
            detective_sigma = params.soph_detective_sigma
            if detective_sigma > 0:
                if raw_likelihoods_A and raw_likelihoods_B:
                    final_A = smooth_likelihood_grid(raw_likelihoods_A, w_t0, detective_sigma)
                    final_B = smooth_likelihood_grid(raw_likelihoods_B, w_t0, detective_sigma)
                    return final_A, final_B
        return raw_likelihoods_A, raw_likelihoods_B


    def compute_predictions(self, final_likelihoods_A: dict, final_likelihoods_B: dict, possible_crumb_coords: list, trial_name: str, agent_type: str, source_data_type: str, mismatched_run: bool) -> tuple[dict, list]:
        predictions = {}
        crumb_data_for_json = []
        for crumb_tuple in possible_crumb_coords:
            likelihood_A = final_likelihoods_A.get(crumb_tuple, 0.0)
            likelihood_B = final_likelihoods_B.get(crumb_tuple, 0.0)
            slider_prediction = normalized_slider_prediction(likelihood_A, likelihood_B)
            predictions[crumb_tuple] = slider_prediction
            
            data_entry = {
                "trial": trial_name, "evidence": "visual", "agent_type": agent_type,
                "crumb_location_world_coords": list(crumb_tuple),
                "slider_prediction": float(slider_prediction),
                "evidence_likelihood_A": float(likelihood_A),
                "evidence_likelihood_B": float(likelihood_B)
            }
            if source_data_type is not None:
                data_entry['source_data_type'] = source_data_type
            if mismatched_run is not None:
                data_entry['mismatched_run'] = mismatched_run
            crumb_data_for_json.append(data_entry)
        return predictions, crumb_data_for_json


    def save_to_json(self, crumb_data_for_json: list, param_log_dir: str, trial_name: str, agent_type: str):
        safe_agent_type_tag = agent_type.replace(" ", "_").replace("/", "_")
        json_filename = os.path.join(param_log_dir, f'detective_preds_{trial_name}_{safe_agent_type_tag}.json')
        with open(json_filename, 'w') as f:
            json.dump(crumb_data_for_json, f, indent=4)


    def simulate_detective(
        self, 
        w_t0: World, 
        trial_name: str, 
        sampled_data_for_detective_calc: dict, # Output from Suspect.simulate_suspect (contains paths for A and B)
        agent_type_being_simulated: str, # 'naive' or 'sophisticated' (or 'uniform') - refers to the type of suspect the detective is modeling
        param_log_dir: str,
        # source_data_type and mismatched_run are for empirical simulator, keep for now
        source_data_type: str = None, 
        mismatched_run: bool = None 
    ) -> tuple[dict, dict, dict]: # (predictions_dict, agent_A_model_output, agent_B_model_output)
                                     # For visual: model_output = likelihood_map
                                     # For audio: model_output = list of sampled audio sequences

        self.logger.info(f"Simulating Detective modeling {agent_type_being_simulated} suspects using {self.params.evidence} evidence.")

        if self.params.evidence == 'visual':
            return self._simulate_detective_visual(
                w_t0, trial_name, sampled_data_for_detective_calc, 
                agent_type_being_simulated, param_log_dir, 
                source_data_type, mismatched_run
            )
        elif self.params.evidence == 'audio':
            return self._simulate_detective_audio(
                w_t0, trial_name, sampled_data_for_detective_calc, 
                agent_type_being_simulated, param_log_dir,
                source_data_type, mismatched_run
            )
        else:
            raise ValueError(f"Unsupported evidence type: {self.params.evidence}")

    def _simulate_detective_visual(
        self, w_t0: World, trial_name: str, sampled_data: dict, 
        agent_type_being_simulated: str, param_log_dir: str,
        source_data_type: str, mismatched_run: bool
    ) -> tuple[dict, dict, dict]:
        
        possible_crumb_coords = w_t0.get_valid_kitchen_crumb_coords_world()
        if not possible_crumb_coords:
            self.logger.warning("No valid crumb coordinates found. Skipping visual detective simulation.")
            return {}, {}, {}

        # sampled_data contains {'A': {path_data_A}, 'B': {path_data_B}}
        # path_data_X contains 'full_sequences', 'middle_sequences', 'chosen_plant_spots'
        raw_likelihoods_A = self._compute_visual_likelihoods_for_agent(w_t0, sampled_data['A'], agent_type_being_simulated, possible_crumb_coords)
        raw_likelihoods_B = self._compute_visual_likelihoods_for_agent(w_t0, sampled_data['B'], agent_type_being_simulated, possible_crumb_coords)

        # Smoothing for visual evidence (consider if this is needed for sophisticated detective's own view)
        # Current soph_detective_sigma applies here.
        final_A_map = raw_likelihoods_A
        final_B_map = raw_likelihoods_B
        if agent_type_being_simulated == 'sophisticated' and self.params.soph_detective_sigma > 0: # Smoothing for detective's own sophisticated model
             self.logger.info(f"Detective smoothing sophisticated suspect likelihood maps with sigma: {self.params.soph_detective_sigma}")
             final_A_map = smooth_likelihood_grid(raw_likelihoods_A, w_t0, self.params.soph_detective_sigma)
             final_B_map = smooth_likelihood_grid(raw_likelihoods_B, w_t0, self.params.soph_detective_sigma)
        elif agent_type_being_simulated == 'naive' and self.params.soph_suspect_sigma > 0: # This sigma is for *suspect's* model of naive detective
            # For the naive detective's actual output, we might not smooth it here unless specified by a different param.
            # The smoothing for the *sophisticated suspect* happens in RSMSimulator.
            # Let's assume naive detective outputs raw likelihoods unless a specific naive_detective_sigma is introduced.
            pass # No smoothing for naive detective's direct output here, smoothing for S Soph is done in RSMSim.

        predictions, crumb_data_for_json = self._compute_visual_predictions_and_json(final_A_map, final_B_map, possible_crumb_coords, 
                                                                            trial_name, agent_type_being_simulated, source_data_type, mismatched_run)
        if param_log_dir: # Only save if param_log_dir is provided
            self.save_visual_to_json(crumb_data_for_json, param_log_dir, trial_name, agent_type_being_simulated, source_data_type, mismatched_run)
        
        return predictions, final_A_map, final_B_map

    def _compute_visual_likelihoods_for_agent(self, w_t0: World, agent_sampled_data: dict, agent_type_being_simulated: str, possible_crumb_coords: list) -> dict:
        likelihood_map = {}
        agent_full_seqs = agent_sampled_data.get('full_sequences', [])
        agent_middle_seqs = agent_sampled_data.get('middle_sequences', [])
        agent_chosen_spots = agent_sampled_data.get('chosen_plant_spots', []) # Used by sophisticated

        if not agent_full_seqs:
            self.logger.warning(f"No full sequences for agent, cannot compute visual likelihoods.")
            return {coord: 0.0 for coord in possible_crumb_coords} # Return zero likelihood for all spots
            
        for crumb_coord in possible_crumb_coords:
            likelihood_map[crumb_coord] = VisualEvidence.get_visual_evidence_likelihood(
                crumb_coord_tuple=crumb_coord,
                agent_full_sequences=agent_full_seqs,
                agent_middle_sequences=agent_middle_seqs,
                world_state=w_t0,
                agent_type_being_simulated=agent_type_being_simulated,
                chosen_plant_spots_for_sequences=agent_chosen_spots
            )
        return likelihood_map

    def _compute_visual_predictions_and_json(self, final_likelihoods_A: dict, final_likelihoods_B: dict, possible_crumb_coords: list, 
                                       trial_name: str, agent_type_being_simulated:str, source_data_type:str=None, mismatched_run:bool=None):
        predictions = {}
        crumb_data_for_json = []
        for coord in possible_crumb_coords:
            l_A = final_likelihoods_A.get(coord, 0.0)
            l_B = final_likelihoods_B.get(coord, 0.0)
            slider_val = normalized_slider_prediction(l_A, l_B)
            predictions[coord] = slider_val
            crumb_data_for_json.append({
                'trial': trial_name,
                'agent_type_simulated': agent_type_being_simulated,
                'source_data_type': source_data_type if mismatched_run else agent_type_being_simulated,
                'mismatched_run': mismatched_run if mismatched_run is not None else False,
                'crumb_x': coord[0],
                'crumb_y': coord[1],
                'L_A': l_A,
                'L_B': l_B,
                'slider': slider_val
            })
        return predictions, crumb_data_for_json

    def save_visual_to_json(self, crumb_data_for_json: list, param_log_dir: str, trial_name:str, agent_type_simulated:str, source_data_type:str=None, mismatched_run:bool=None):
        model_condition_name = agent_type_simulated
        if mismatched_run:
            model_condition_name = f"{agent_type_simulated}_as_{source_data_type}"
        
        output_filename = os.path.join(param_log_dir, f"{trial_name}_{model_condition_name}_visual_crumb_predictions.json")
        try:
            serializable_data = ensure_serializable(crumb_data_for_json)
            with open(output_filename, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            self.logger.info(f"Saved visual predictions to {output_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save visual predictions to {output_filename}: {e}")

    def _simulate_detective_audio(
        self, w_t0: World, trial_name: str, sampled_data_for_detective_calc: dict, 
        agent_type_being_simulated: str, param_log_dir: str, 
        source_data_type: str, mismatched_run: bool
    ) -> tuple[dict, tuple[list, list], tuple[list, list]]: # (predictions_dict, model_A_output, model_B_output)
                                     # model_output is (list_of_to_fridge_steps, list_of_from_fridge_steps)

        self.logger.info(f"Simulating Detective, modeling {agent_type_being_simulated} suspects, using audio evidence. Input data from {source_data_type} suspects.")
        
        agent_A_full_compressed_audios = sampled_data_for_detective_calc['A'].get('audio_sequences', [])
        agent_B_full_compressed_audios = sampled_data_for_detective_calc['B'].get('audio_sequences', [])

        # Always build the "model output" which consists of lists of to/from fridge steps
        # This is used by RSMSimulator to populate params for sophisticated suspect if agent_type_being_simulated == 'naive'
        model_A_to_fridge_steps = []
        model_A_from_fridge_steps = []
        for audio_seq in agent_A_full_compressed_audios:
            if audio_seq and len(audio_seq) == 5 and isinstance(audio_seq[0], int) and isinstance(audio_seq[4], int):
                model_A_to_fridge_steps.append(audio_seq[0])
                model_A_from_fridge_steps.append(audio_seq[4])
            else:
                self.logger.warning(f"Malformed audio sequence for A for model building/prediction: {audio_seq} (type: {agent_type_being_simulated}, source: {source_data_type})")
        
        model_B_to_fridge_steps = []
        model_B_from_fridge_steps = []
        for audio_seq in agent_B_full_compressed_audios:
            if audio_seq and len(audio_seq) == 5 and isinstance(audio_seq[0], int) and isinstance(audio_seq[4], int):
                model_B_to_fridge_steps.append(audio_seq[0])
                model_B_from_fridge_steps.append(audio_seq[4])
            else:
                self.logger.warning(f"Malformed audio sequence for B for model building/prediction: {audio_seq} (type: {agent_type_being_simulated}, source: {source_data_type})")

        model_A_output = (model_A_to_fridge_steps, model_A_from_fridge_steps)
        model_B_output = (model_B_to_fridge_steps, model_B_from_fridge_steps)
        self.logger.info(f"Extracted audio step models for {agent_type_being_simulated} detective: "
                         f"A_to({len(model_A_to_fridge_steps)}), A_from({len(model_A_from_fridge_steps)}); "
                         f"B_to({len(model_B_to_fridge_steps)}), B_from({len(model_B_from_fridge_steps)})")

        # Always attempt to generate predictions
        audio_predictions = {}
        audio_data_for_json = []

        if not agent_A_full_compressed_audios and not agent_B_full_compressed_audios:
            self.logger.warning(f"No sampled audio sequences available for A or B for {agent_type_being_simulated} detective. Cannot make predictions.")
            return audio_predictions, model_A_output, model_B_output
        
        ground_truth_audios = generate_ground_truth_audio_sequences(w_t0, self.params)
        if not ground_truth_audios:
            self.logger.warning(f"No ground truth audio sequences generated for {agent_type_being_simulated} detective. Cannot make audio predictions.")
            return audio_predictions, model_A_output, model_B_output

        sigma_factor = self.params.audio_segment_similarity_sigma

        for gt_audio_idx, gt_audio_seq in enumerate(ground_truth_audios):
            if not (isinstance(gt_audio_seq, list) and len(gt_audio_seq) == 5 and isinstance(gt_audio_seq[0], int) and isinstance(gt_audio_seq[4], int)):
                self.logger.warning(f"Skipping malformed ground truth audio: {gt_audio_seq}")
                continue
            gt_steps_to = gt_audio_seq[0]
            gt_steps_from = gt_audio_seq[4]

            # Likelihood that gt_audio_seq was generated by Agent A
            L_gt_given_A = 0.0
            if agent_A_full_compressed_audios: # Use all valid audios from A
                path_likelihoods_A = []
                for a_path_audio in agent_A_full_compressed_audios: # These are the original full compressed audios
                    if not (isinstance(a_path_audio, list) and len(a_path_audio) == 5 and isinstance(a_path_audio[0], int) and isinstance(a_path_audio[4], int)):
                        # Already warned during model building, but double check here if flow changes
                        continue 
                    if gt_audio_seq[1:4] != a_path_audio[1:4]: # Check non-step events
                        path_likelihoods_A.append(0.0)
                        continue
                    
                    path_steps_to_A = a_path_audio[0]
                    path_steps_from_A = a_path_audio[4]
                    
                    lik_to_A = single_segment_audio_likelihood(gt_steps_to, path_steps_to_A, sigma_factor)
                    lik_from_A = single_segment_audio_likelihood(gt_steps_from, path_steps_from_A, sigma_factor)
                    path_likelihoods_A.append(lik_to_A * lik_from_A)
                if path_likelihoods_A:
                    L_gt_given_A = np.mean(path_likelihoods_A)
            
            # Likelihood that gt_audio_seq was generated by Agent B
            L_gt_given_B = 0.0
            if agent_B_full_compressed_audios: # Use all valid audios from B
                path_likelihoods_B = []
                for b_path_audio in agent_B_full_compressed_audios:
                    if not (isinstance(b_path_audio, list) and len(b_path_audio) == 5 and isinstance(b_path_audio[0], int) and isinstance(b_path_audio[4], int)):
                        continue
                    if gt_audio_seq[1:4] != b_path_audio[1:4]: # Check non-step events
                        path_likelihoods_B.append(0.0)
                        continue

                    path_steps_to_B = b_path_audio[0]
                    path_steps_from_B = b_path_audio[4]

                    lik_to_B = single_segment_audio_likelihood(gt_steps_to, path_steps_to_B, sigma_factor)
                    lik_from_B = single_segment_audio_likelihood(gt_steps_from, path_steps_from_B, sigma_factor)
                    path_likelihoods_B.append(lik_to_B * lik_from_B)
                if path_likelihoods_B:
                    L_gt_given_B = np.mean(path_likelihoods_B)

            slider_val = normalized_slider_prediction(L_gt_given_A, L_gt_given_B)
            audio_predictions[f"gt_audio_{gt_audio_idx}"] = slider_val
            
            audio_data_for_json.append({
                'trial': trial_name,
                'agent_type_simulated': agent_type_being_simulated, # This is the type of detective (naive/soph)
                'source_data_type': source_data_type, # This is the type of suspect paths used to make this prediction
                'mismatched_run': mismatched_run if mismatched_run is not None else False,
                'ground_truth_audio_id': f"gt_audio_{gt_audio_idx}",
                'ground_truth_audio_sequence': gt_audio_seq,
                'L_A_given_gt': L_gt_given_A, 
                'L_B_given_gt': L_gt_given_B, 
                'slider': slider_val
            })

        if param_log_dir and audio_data_for_json : # Save if there's data
            self.logger.info(f"Attempting to save audio detective predictions for trial {trial_name}, detective type: {agent_type_being_simulated}, source data: {source_data_type}")
            self.save_audio_to_json(audio_data_for_json, param_log_dir, trial_name, agent_type_being_simulated, source_data_type, mismatched_run)
        elif not audio_data_for_json:
             self.logger.info(f"No audio prediction data to save for trial {trial_name}, detective type: {agent_type_being_simulated}")


        return audio_predictions, model_A_output, model_B_output

    def save_audio_to_json(self, audio_data_for_json: list, param_log_dir: str, trial_name:str, agent_type_simulated:str, source_data_type:str=None, mismatched_run:bool=None):
        model_condition_name = agent_type_simulated
        if mismatched_run:
            model_condition_name = f"{agent_type_simulated}_as_{source_data_type}"
        
        output_filename = os.path.join(param_log_dir, f"{trial_name}_{model_condition_name}_audio_predictions.json")
        try:
            serializable_data = ensure_serializable(audio_data_for_json)
            with open(output_filename, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            self.logger.info(f"Saved audio predictions to {output_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save audio predictions to {output_filename}: {e}")
