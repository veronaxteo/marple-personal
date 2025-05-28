import numpy as np
from scipy.stats import norm
import networkx as nx
from igraph import Graph
import logging
from typing import TYPE_CHECKING

# from test_world import World
from params import SimulationParams

if TYPE_CHECKING:
    from test_world import World


def get_audio_tokens_for_path(world_state: 'World', path_coords: list) -> list:
    """Generates a raw list of audio tokens for an agent traversing a path."""
    raw_tokens = []
    if not path_coords: return raw_tokens

    fridge_access_point = world_state.get_fridge_access_point()
    fridge_event_added = False
    for i, coord in enumerate(path_coords):
        if i > 0:
            raw_tokens.append('step')

        if coord == fridge_access_point and not fridge_event_added:
            raw_tokens.extend(['fridge_opened', 'snack_picked_up', 'fridge_closed'])
            fridge_event_added = True
    return raw_tokens


def parse_raw_audio_tokens(raw_audio_tokens: list) -> list:
    """
    Compresses sequences of raw tokens.
    Returns a list of [num_steps_to, <fridge_events>, num_steps_from]
    """
    compressed_tokens = []
    current_step_count = 0
    for token in raw_audio_tokens:
        if token == 'step':
            current_step_count += 1
        else:
            if current_step_count > 0:
                compressed_tokens.append(current_step_count)
            compressed_tokens.append(token)
            current_step_count = 0
    if current_step_count > 0:
        compressed_tokens.append(current_step_count)
    return compressed_tokens
    

def get_compressed_audio_from_path(world_state: 'World', path_coords: list) -> list:
    """Converts a world coordinate path to a compressed audio token sequence."""
    raw_tokens = get_audio_tokens_for_path(world_state, path_coords)
    # print(f"Raw tokens: {raw_tokens}")
    compressed_tokens = parse_raw_audio_tokens(raw_tokens)
    # print(f"Compressed tokens: {compressed_tokens}")
    return compressed_tokens


def single_segment_audio_likelihood(gt_steps: int, path_steps: int, sigma_factor: float = 0.1) -> float:
    """
    Computes the likelihood of observing path_steps for a segment,
    given an expected gt_steps for that segment.
    Uses a Gaussian PDF, normalized by the PDF's peak value.
    """
    if not isinstance(gt_steps, (int, float)) or not isinstance(path_steps, (int, float)):
        logging.getLogger(__name__).warning(f"Invalid step types for single_segment_audio_likelihood: gt_steps={gt_steps} (type {type(gt_steps)}), path_steps={path_steps} (type {type(path_steps)})")
        return 0.0

    sigma = max(1.0, gt_steps * sigma_factor) if gt_steps > 0 else 1.0
    
    # Calculate likelihood of path_steps
    lik = norm.pdf(path_steps, loc=gt_steps, scale=sigma)
    
    # Calculate the maximum possible likelihood for this gt_steps (i.e., when path_steps == gt_steps)
    # This is used for normalization, so likelihood is between 0 and 1.
    max_lik = norm.pdf(gt_steps, loc=gt_steps, scale=sigma)
    
    if max_lik == 0: # Avoid division by zero; can happen if sigma is extremely small and gt_steps is far from path_steps
        # If max_lik is 0, and lik is also 0, then they are "equally unlikely" in a sense (normalized could be 1 if lik==max_lik).
        # If lik is non-zero (highly unlikely if max_lik is 0 unless sigma is 0), it's an anomaly.
        # A robust approach: if max_lik is 0, it means gt_steps itself has 0 probability under the distribution centered at itself,
        # which implies a degenerate case (e.g. sigma became 0, or an underflow).
        # If lik is also 0, then 0/0 -> conventionally 1 if it's the same zero, or 0 if path_steps != gt_steps.
        # Let's be conservative: if max_lik is zero, only return 1.0 if lik is also zero AND path_steps perfectly matches gt_steps.
        # Otherwise, it's effectively zero likelihood.
        # However, norm.pdf(x, loc=x, scale=positive_sigma) should always be > 0.
        # So max_lik should only be 0 if sigma somehow became non-positive or due to extreme underflow.
        # Given sigma = max(1.0, ...), sigma is always >= 1.0. So max_lik should always be positive.
        # This case might be more for theoretical robustness.
        return 1.0 if lik == 0 and path_steps == gt_steps else 0.0

    normalized_lik = lik / max_lik
    return normalized_lik


def get_segmented_audio_likelihood(ground_truth_compressed_tokens: list, path_compressed_tokens: list, sigma_factor: float = 0.1) -> float:
    """
    DEPRECATED or to be REFACTORED.
    Original function to compute likelihood of path audio given ground truth (gt) audio.
    This function's logic will be replaced by direct calls to single_segment_audio_likelihood
    for 'to_fridge' and 'from_fridge' step components within the agent/world logic.
    """
    logger = logging.getLogger(__name__)
    logger.warning("get_segmented_audio_likelihood is being called but is slated for deprecation/refactoring.")

    # Basic validation (assuming 5-element structure for now, though this will change)
    if not (len(ground_truth_compressed_tokens) == 5 and len(path_compressed_tokens) == 5):
        # logger.debug(f"Mismatched token list lengths: GT: {ground_truth_compressed_tokens}, Path: {path_compressed_tokens}")
        return 0.0 
    
    # Check if fridge events match (elements 1, 2, 3)
    if ground_truth_compressed_tokens[1:4] != path_compressed_tokens[1:4]:
        # logger.debug(f"Non-matching fridge events: GT: {ground_truth_compressed_tokens[1:4]}, Path: {path_compressed_tokens[1:4]}")
        return 0.0

    gt_steps_to = ground_truth_compressed_tokens[0]
    gt_steps_from = ground_truth_compressed_tokens[4] # Last element for 5-element list
    path_steps_to = path_compressed_tokens[0]
    path_steps_from = path_compressed_tokens[4] # Last element

    # Ensure step counts are numbers
    if not all(isinstance(s, (int, float)) for s in [gt_steps_to, gt_steps_from, path_steps_to, path_steps_from]):
        logger.warning(f"Non-numeric step count found. GT_to:{gt_steps_to}, GT_from:{gt_steps_from}, Path_to:{path_steps_to}, Path_from:{path_steps_from}")
        return 0.0

    lik_to = single_segment_audio_likelihood(gt_steps_to, path_steps_to, sigma_factor)
    lik_from = single_segment_audio_likelihood(gt_steps_from, path_steps_from, sigma_factor)
    
    final_likelihood = lik_to * lik_from # Combine segment likelihoods by multiplication
    return final_likelihood


def generate_ground_truth_audio_sequences(world: 'World', params: SimulationParams) -> list:
    """
    Generates a list of 'ground truth' compressed audio sequences.
    These represent a spectrum of possible audio events the detective might expect.
    Example: [[6, 'fridge_opened', ..., 6], [8, 'fridge_opened', ..., 6], ...]
    """
    logger = logging.getLogger(__name__)
    ground_truths = []
    fridge_access_point = world.get_fridge_access_point()
    if not fridge_access_point:
        logger.error("Cannot generate ground truth audio: Fridge access point not found.")
        return []

    start_A_coord = world.start_coords['A']
    start_B_coord = world.start_coords['B']

    # Convert coordinates to VIDs for igraph
    start_A_vid = world.node_to_vid.get(start_A_coord)
    start_B_vid = world.node_to_vid.get(start_B_coord)
    fridge_vid = world.node_to_vid.get(fridge_access_point)

    if start_A_vid is None:
        logger.error(f"Start A coordinate {start_A_coord} not in node_to_vid map.")
        return [] # Or handle error as appropriate
    if start_B_vid is None:
        logger.error(f"Start B coordinate {start_B_coord} not in node_to_vid map.")
        return []
    if fridge_vid is None:
        logger.error(f"Fridge access point {fridge_access_point} not in node_to_vid map.")
        return []

    path_A_to_fridge = float('inf')
    try:
        # path_A_to_fridge = nx.shortest_path_length(world.graph, source=start_A_coord, target=fridge_access_point)
        path_len_matrix = world.igraph.shortest_paths(source=start_A_vid, target=fridge_vid)
        if path_len_matrix and path_len_matrix[0]:
            path_A_to_fridge = path_len_matrix[0][0]
    except Exception as e: # Catch generic igraph errors or if target is not reachable
        logger.warning(f"Could not find path from start_A_vid {start_A_vid} to fridge_vid {fridge_vid}: {e}")
        pass # path_A_to_fridge remains float('inf')

    path_B_to_fridge = float('inf')
    try:
        # path_B_to_fridge = nx.shortest_path_length(world.graph, source=start_B_coord, target=fridge_access_point)
        path_len_matrix = world.igraph.shortest_paths(source=start_B_vid, target=fridge_vid)
        if path_len_matrix and path_len_matrix[0]:
            path_B_to_fridge = path_len_matrix[0][0]
    except Exception as e:
        logger.warning(f"Could not find path from start_B_vid {start_B_vid} to fridge_vid {fridge_vid}: {e}")
        pass

    if path_A_to_fridge == float('inf') and path_B_to_fridge == float('inf'):
        logger.warning("Neither agent can reach the fridge (igraph). No audio ground truths generated.")
        return []

    min_steps_to_fridge_overall = min(path_A_to_fridge, path_B_to_fridge)
    closer_agent_start_vid = start_A_vid if path_A_to_fridge <= path_B_to_fridge else start_B_vid
    # closer_agent_start_coord = start_A_coord if path_A_to_fridge <= path_B_to_fridge else start_B_coord

    min_steps_fridge_to_closer_agent_start = float('inf')
    try:
        # min_steps_fridge_to_closer_agent_start = nx.shortest_path_length(world.graph, source=fridge_access_point, target=closer_agent_start_coord)
        path_len_matrix = world.igraph.shortest_paths(source=fridge_vid, target=closer_agent_start_vid)
        if path_len_matrix and path_len_matrix[0]:
            min_steps_fridge_to_closer_agent_start = path_len_matrix[0][0]
    except Exception as e:
        logger.warning(f"Cannot find path from fridge_vid {fridge_vid} to closer_agent_start_vid {closer_agent_start_vid}: {e}. Using steps_to_fridge for steps_from.")
        min_steps_fridge_to_closer_agent_start = min_steps_to_fridge_overall


    base_steps_to = int(min_steps_to_fridge_overall) if min_steps_to_fridge_overall != float('inf') else params.max_steps +1 # Ensure invalid paths don't generate GTs
    base_steps_from = int(min_steps_fridge_to_closer_agent_start) if min_steps_fridge_to_closer_agent_start != float('inf') else params.max_steps +1
    
    fridge_events = ['fridge_opened', 'snack_picked_up', 'fridge_closed']

    # Generate variations
    # Iterate through increments for steps_to_fridge
    for inc_to in range(0, params.max_steps + 1, params.audio_gt_step_size):
        current_steps_to = base_steps_to + inc_to
        if current_steps_to > params.max_steps : continue

        # Iterate through increments for steps_from_fridge
        for inc_from in range(0, params.max_steps + 1, params.audio_gt_step_size):
            current_steps_from = base_steps_from + inc_from
            if current_steps_from > params.max_steps: continue
            
            gt_sequence = [current_steps_to] + fridge_events + [current_steps_from]
            if gt_sequence not in ground_truths: # Avoid duplicates
                ground_truths.append(gt_sequence)
                
    if not ground_truths and (base_steps_to <= params.max_steps and base_steps_from <= params.max_steps):
         # Add at least the base case if no increments were made but it's valid
        ground_truths.append([base_steps_to] + fridge_events + [base_steps_from])

    logger.info(f"Generated {len(ground_truths)} ground truth audio sequences.")

    return sorted(ground_truths)


class Evidence:
    """Base evidence class. (Now primarily a namespace for static methods or future expansion)"""
    pass # Base class can be minimal or abstract if all current types are very different


class VisualEvidence(Evidence):
    """Visual evidence class (crumb-based)."""
    def __init__(self, **kwargs): # Keep init if it's used elsewhere, otherwise can be removed
        super().__init__(**kwargs)

    @staticmethod
    def get_visual_evidence_likelihood(
        crumb_coord_tuple,
        agent_full_sequences,
        agent_middle_sequences,
        world_state: 'World',
        agent_type_being_simulated='naive',
        chosen_plant_spots_for_sequences=None
    ):
        """
        Calculates the likelihood of observing a crumb at crumb_coord_tuple.
        (Method moved to be static as it doesn't rely on VisualEvidence instance state)
        """
        logger = logging.getLogger(__name__)
        total_likelihood = 0.0
        num_sequences = len(agent_full_sequences)
        if num_sequences == 0: return 0.0

        fridge_access_point = world_state.get_fridge_access_point()
        initial_door_states = world_state.get_initial_door_states() # This likely needs igraph if doors are dynamic
                                                                  # For now, assuming it returns a simple dict not needing graph state beyond init.
        middle_sequence_lengths = [len(seq) if seq else 1 for seq in agent_middle_sequences]

        for i, sequence in enumerate(agent_full_sequences):
            current_middle_len = middle_sequence_lengths[i]
            if current_middle_len == 0: current_middle_len = 1

            likelihood_for_sequence = 0.0
            if agent_type_being_simulated == 'sophisticated':
                if chosen_plant_spots_for_sequences is None or len(chosen_plant_spots_for_sequences) != num_sequences:
                    logger.warning("Chosen plant spots not provided or mismatched for sophisticated agent; defaulting to 0 likelihood for path.")
                else:
                    chosen_plant_spot = chosen_plant_spots_for_sequences[i]
                    if chosen_plant_spot is not None and crumb_coord_tuple == chosen_plant_spot:
                        likelihood_for_sequence = 1.0
            else: # naive or uniform
                simulated_door_states = initial_door_states.copy()
                on_return = False
                generated_crumbs_for_this_path = set()
                if fridge_access_point is None and agent_type_being_simulated != 'uniform':
                    logger.debug("Fridge access point is None, cannot simulate crumb generation for naive agent.")
                
                for coord_idx, coord in enumerate(sequence):
                    # node_data = world_state.graph.nodes.get(coord, {}) # NX version
                    if coord in world_state.node_to_vid:
                        vid = world_state.node_to_vid[coord]
                        node_attrs = world_state.igraph.vs[vid]
                        is_kitchen = node_attrs['room'] == 'Kitchen'
                        is_door = node_attrs['is_door']
                    else:
                        # If coord is not in node_to_vid, it's not a valid graph node (e.g. furniture, outside map)
                        # This path is likely invalid or goes through non-navigable space.
                        # For crumb generation, this means a crumb cannot be dropped at 'coord' by this logic.
                        # Default to not kitchen, not door if not a graph node for safety, though path should ideally only contain graph nodes.
                        logger.warning(f"Coordinate {coord} in sequence not found in world_state.node_to_vid map during visual evidence calculation.")
                        is_kitchen = False
                        is_door = False 

                    if fridge_access_point and coord == fridge_access_point:
                        on_return = True
                    
                    if is_door and coord in simulated_door_states: # simulated_door_states keys are coord tuples
                        simulated_door_states[coord] = 'open'

                    if on_return and is_kitchen and not is_door: # Crumb dropped in kitchen on return path
                        generated_crumbs_for_this_path.add(coord)
                
                if crumb_coord_tuple in generated_crumbs_for_this_path:
                    likelihood_for_sequence = 1.0 / current_middle_len
            
            total_likelihood += likelihood_for_sequence
        
        average_likelihood = total_likelihood / num_sequences
        return average_likelihood


class AudioEvidence(Evidence): # This class might just become a namespace for audio-related static methods
    """Audio evidence class. (Primarily a namespace for audio processing utilities)"""
    # No constructor needed if all methods are static or belong at module level
    pass
    # All methods (get_audio_tokens_for_path, parse_raw_audio_tokens, get_segmented_audio_likelihood)
    # have been moved to module level or could be static methods here.
    # For consistency, one could make them static methods of AudioEvidence,
    # but module-level functions are also fine in Python for utilities.
