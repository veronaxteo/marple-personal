import numpy as np
import json
import os

from world import World
from params import SimulationParams
from evidence import VisualEvidence, AudioEvidence
from utils import normalized_slider_prediction, smooth_likelihood_grid


class Agent():
    """
    The base agent class.

    Attributes:
        id (str): agent id (e.g., 'A', 'B')
        data_type (str): type of data to use for evidence (e.g., visual, audio, multimodal)
        params (dict): parameters for the agent, including agent type (e.g., naive, sophisticated), w, temp, etc.
    """
    def __init__(self, id, data_type, params):
        self.id = id
        self.data_type = data_type
        self.params = params


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


    def simulate_suspect(self, w_t0: World, 
                        simple_paths_A_seqs: list, simple_paths_B_seqs: list,
                        agent_type: str, num_sample_paths: int, params: SimulationParams) -> dict:
        """Simulates suspect paths by sampling from simple paths according to the agents' cost functions."""
        sampled_data = {}
        for agent in ['A', 'B']:
            simple_sequences = simple_paths_A_seqs if agent == 'A' else simple_paths_B_seqs

            call_args = {
                'agent_id': agent,
                'simple_path_sequences': simple_sequences,
                'num_sample_paths': num_sample_paths,
                'agent_type': agent_type,
                'naive_A_crumb_likelihoods_map': None,
                'naive_B_crumb_likelihoods_map': None,
                'w': 0.0,
                'temp': 0.0,
                'noisy_planting_sigma': 0.0
            }

            if agent_type == 'sophisticated':
                call_args['w'] = params.w if params.w is not None else 0.0
                call_args['temp'] = params.s_temp if params.s_temp is not None else 0.0
                call_args['naive_A_crumb_likelihoods_map'] = params.naive_A_crumb_likelihoods_map
                call_args['naive_B_crumb_likelihoods_map'] = params.naive_B_crumb_likelihoods_map
                call_args['noisy_planting_sigma'] = params.noisy_planting_sigma if params.noisy_planting_sigma is not None else 0.0
            elif agent_type == 'naive':
                call_args['w'] = params.w if params.w is not None else 0.0
                call_args['temp'] = params.n_temp if params.n_temp is not None else 0.0

            result = w_t0.get_sample_paths(**call_args)
            sampled_data[agent] = result
            
        return sampled_data


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
                    door_close_prob=params.door_close_prob,
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
            sigma = params.soph_detective_sigma
            if sigma > 0:
                if raw_likelihoods_A and raw_likelihoods_B:
                    final_A = smooth_likelihood_grid(raw_likelihoods_A, w_t0, sigma)
                    final_B = smooth_likelihood_grid(raw_likelihoods_B, w_t0, sigma)
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


    def simulate_detective(self, w_t0: World,
                           trial_name: str, sampled_data: dict, agent_type: str, params: SimulationParams, param_log_dir: str,
                           source_data_type: str=None, mismatched_run: bool=None) -> tuple[dict, dict, dict]:
        possible_crumb_coords = w_t0.get_valid_kitchen_crumb_coords_world()

        raw_likelihoods_A, raw_likelihoods_B = self.compute_likelihoods(w_t0, sampled_data, agent_type, params, possible_crumb_coords)

        final_likelihoods_A, final_likelihoods_B = self.smooth_likelihoods(raw_likelihoods_A, raw_likelihoods_B, agent_type, w_t0, params)

        predictions, crumb_data_for_json = self.compute_predictions(final_likelihoods_A, final_likelihoods_B, possible_crumb_coords, 
                                                                    trial_name, agent_type, source_data_type, mismatched_run)

        self.save_to_json(crumb_data_for_json, param_log_dir, trial_name, agent_type)

        return predictions, final_likelihoods_A, final_likelihoods_B
