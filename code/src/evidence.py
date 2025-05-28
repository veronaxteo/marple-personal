import logging
import numpy as np
from scipy.stats import norm


class Evidence:
    """
    Base evidence class.

    Attributes:
        world_state (World): world state
        params (dict): parameters for the evidence
    """
    world_state = None
    params = None


class VisualEvidence(Evidence):
    """
    Visual evidence class.

    Attributes:
        world_state (inherited): World, world state
        params (inherited): dict, parameters for the evidence
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def get_visual_evidence_likelihood(
        crumb_coord_tuple, 
        agent_full_sequences, 
        agent_middle_sequences, 
        world_state, 
        agent_type_being_simulated='naive',  
        chosen_plant_spots_for_sequences=None
    ):
        """
        Calculates the likelihood of observing a crumb at crumb_coord_tuple, given the agent's sampled full path sequences (world coords).
        This involves simulating each full path to determine where crumbs would drop.
        For 'naive'/'uniform': Likelihood for a path is 1 / len(middle_path_segment) if the simulation path generates the crumb, 0 otherwise.
        For 'sophisticated': Likelihood is 1.0 if crumb_coord_tuple matches the chosen_plant_spot for that path, 0 otherwise.
        The final likelihood is averaged over all sampled paths.
        """

        total_likelihood = 0.0
        num_sequences = len(agent_full_sequences)
            
        fridge_access_point = world_state.get_fridge_access_point()
        initial_door_states = world_state.get_initial_door_states()

        middle_sequence_lengths = [len(seq) if seq else 1 for seq in agent_middle_sequences] # ensure no div by zero

        for i, sequence in enumerate(agent_full_sequences):
            current_middle_len = middle_sequence_lengths[i]
            if current_middle_len == 0: current_middle_len = 1 

            likelihood_for_sequence = 0.0 

            if agent_type_being_simulated == 'sophisticated':
                if chosen_plant_spots_for_sequences is None or len(chosen_plant_spots_for_sequences) != num_sequences:
                    logger = logging.getLogger(__name__)
                    logger.warning("Chosen plant spots not provided or mismatched for sophisticated agent; defaulting to 0 likelihood for path.")

                else:
                    chosen_plant_spot = chosen_plant_spots_for_sequences[i]
                    if chosen_plant_spot is not None and crumb_coord_tuple == chosen_plant_spot:
                        likelihood_for_sequence = 1.0
                    else:
                        likelihood_for_sequence = 0.0
            else:
                simulated_door_states = initial_door_states.copy()
                on_return = False
                generated_crumbs_for_this_path = set()
                
                # Simulate full path
                for coord in sequence:
                    node_data = world_state.graph.nodes.get(coord, {})
                    is_kitchen = node_data.get('room') == 'Kitchen'
                    is_door = node_data.get('is_door', False)

                    if coord == fridge_access_point:
                        on_return = True

                    # Simulate door
                    if is_door and coord in simulated_door_states:
                        simulated_door_states[coord] = 'open'
                        # if random.random() < door_close_prob:
                        #     simulated_door_states[coord] = 'closed'

                    # Track generated crumbs
                    if on_return:
                        if is_kitchen and not is_door:
                            generated_crumbs_for_this_path.add(coord)

                if crumb_coord_tuple in generated_crumbs_for_this_path:
                    # Likelihood is inverse of the path length from fridge --> door
                    likelihood_for_sequence = 1.0 / current_middle_len
            
            total_likelihood += likelihood_for_sequence

        average_likelihood = total_likelihood / num_sequences
        return average_likelihood


class AudioEvidence(Evidence):
    """
    Audio evidence class.

    Attributes:
        world_state (inherited): World, world state
        params (inherited): dict, parameters for the evidence
        raw_audio_tokens (list): raw audio tokens, e.g. ['step', 'step', 'step', 'fridge_opened', 'snack_picked_up', 'fridge_closed', 'step', 'step', 'step']
        compressed_audio_tokens (list): compressed audio tokens, e.g. [3, 'fridge_opened', 'snack_picked_up', 'fridge_closed', 3]
    """
    def __init__(self, audio_tokens):
        super().__init__()
        self.raw_audio_tokens = list(audio_tokens)
        self.compressed_audio_tokens = []
        self.parse_audio_tokens()


    def parse_audio_tokens(self):
        """Compresses sequences of tokens and identifies key events."""
        tokens = self.raw_audio_tokens
        temp_parsed_tokens = []
        current_step_count = 0

        # Group consecutive steps and keep discrete events
        # TODO: make more concise
        for token in tokens:
            if token == 'step':
                current_step_count += 1
            else:
                if current_step_count > 0:
                    temp_parsed_tokens.append(current_step_count)
                temp_parsed_tokens.append(token)
                current_step_count = 0
        if current_step_count > 0:  # add any trailing steps
            temp_parsed_tokens.append(current_step_count)
        
        # Ensure specific [steps, event, event, event, steps] structure
        final_compressed = []
        
        # 1. Steps to fridge
        if temp_parsed_tokens and isinstance(temp_parsed_tokens[0], int):
            final_compressed.append(temp_parsed_tokens.pop(0))

        # 2. Fridge events
        expected_fridge_events = ['fridge_opened', 'snack_picked_up', 'fridge_closed']
        for _, expected_event in enumerate(expected_fridge_events):
            if temp_parsed_tokens and temp_parsed_tokens[0] == expected_event:
                final_compressed.append(temp_parsed_tokens.pop(0))
            else:
                print(f"Missing expected event: {expected_event}")
                exit()

        # 3. Steps from fridge
        if temp_parsed_tokens and isinstance(temp_parsed_tokens[0], int):
            final_compressed.append(temp_parsed_tokens.pop(0))
            
        self.compressed_audio_tokens = final_compressed


    def get_audio_tokens_for_path(world_state, path_coords):
        """Generates a raw list of audio tokens for an agent traversing a path."""
        raw_tokens = []
        fridge_access_point = world_state.get_fridge_access_point() 

        fridge_event_added = False
        for i, coord in enumerate(path_coords):
            if i > 0: 
                raw_tokens.append('step')

            if coord == fridge_access_point and not fridge_event_added:
                raw_tokens.extend(['fridge_opened', 'snack_picked_up', 'fridge_closed'])
                fridge_event_added = True
            
        return raw_tokens


    def get_segmented_audio_likelihood(ground_truth_compressed_tokens, path_compressed_tokens, sigma_factor=0.2):
        """Computes likelihood of path audio given ground truth (gt) audio."""
        # TODO: currently hardcoded (will need to change if we add more events/sounds)
        gt_steps_to = ground_truth_compressed_tokens.pop(0)
        gt_steps_from = ground_truth_compressed_tokens[-1]
        path_steps_to = path_compressed_tokens.pop(0)
        path_steps_from = path_compressed_tokens[-1]

        segment_likelihoods = []

        # TODO: will need to revise sigma factor / gaussian log
        # Likelihood for steps to fridge
        sigma_to = max(1.0, gt_steps_to * sigma_factor) if gt_steps_to > 0 else 1.0
        lik_to = norm.pdf(path_steps_to, loc=gt_steps_to, scale=sigma_to)
        max_lik_to = norm.pdf(gt_steps_to, loc=gt_steps_to, scale=sigma_to) if gt_steps_to > 0 else norm.pdf(0, loc=0, scale=1.0) 
        segment_likelihoods.append(lik_to / max_lik_to if max_lik_to > 0 else (1.0 if gt_steps_to == path_steps_to else 0.0))

        # Likelihood for steps from fridge
        sigma_from = max(1.0, gt_steps_from * sigma_factor) if gt_steps_from > 0 else 1.0
        lik_from = norm.pdf(path_steps_from, loc=gt_steps_from, scale=sigma_from)
        max_lik_from = norm.pdf(gt_steps_from, loc=gt_steps_from, scale=sigma_from) if gt_steps_from > 0 else norm.pdf(0, loc=0, scale=1.0)
        segment_likelihoods.append(lik_from / max_lik_from if max_lik_from > 0 else (1.0 if gt_steps_from == path_steps_from else 0.0))
        
        final_likelihood = np.prod(segment_likelihoods)
        return final_likelihood
