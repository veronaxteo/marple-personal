import logging
from scipy.stats import norm

# TODO: remove TYPE_CHECKING and fix imports
from typing import TYPE_CHECKING
from params import SimulationParams

if TYPE_CHECKING:
    from world import World


class Evidence:
    """Base evidence class for future expansion"""
    pass


class VisualEvidence(Evidence):
    """Visual evidence processing utilities"""
    
    @staticmethod
    def get_visual_evidence_likelihood(*args, **kwargs):
        """Static method wrapper for visual evidence likelihood calculation"""
        return get_visual_evidence_likelihood(*args, **kwargs)


class AudioEvidence(Evidence):
    """Audio evidence processing utilities"""
    
    @staticmethod
    def single_segment_audio_likelihood(*args, **kwargs):
        """Static method wrapper for single segment audio likelihood"""
        return single_segment_audio_likelihood(*args, **kwargs)
    
    @staticmethod
    def generate_ground_truth_sequences(*args, **kwargs):
        """Static method wrapper for ground truth audio sequence generation"""
        return generate_ground_truth_audio_sequences(*args, **kwargs)
    

# Visual functions

def get_visual_evidence_likelihood(crumb_coord_tuple, agent_full_sequences, agent_middle_sequences,
                                 world_state: 'World', agent_type_being_simulated='naive',
                                 chosen_plant_spots_for_sequences=None):
    """
    Calculate likelihood of observing a crumb at given coordinates.
    Handles both naive (random crumb dropping) and sophisticated (strategic planting) agents.
    """
    logger = logging.getLogger(__name__)
    total_likelihood = 0.0
    num_sequences = len(agent_full_sequences)
    
    if num_sequences == 0:
        return 0.0

    fridge_access_point = world_state.get_fridge_access_point()
    initial_door_states = world_state.get_initial_door_states()
    middle_sequence_lengths = [len(seq) if seq else 1 for seq in agent_middle_sequences]

    for i, sequence in enumerate(agent_full_sequences):
        current_middle_len = middle_sequence_lengths[i]
        if current_middle_len == 0:
            current_middle_len = 1

        likelihood_for_sequence = 0.0
        
        if agent_type_being_simulated == 'sophisticated':
            # Sophisticated agents plant crumbs strategically
            if (chosen_plant_spots_for_sequences is not None and 
                len(chosen_plant_spots_for_sequences) == num_sequences):
                chosen_plant_spot = chosen_plant_spots_for_sequences[i]
                if chosen_plant_spot is not None and crumb_coord_tuple == chosen_plant_spot:
                    likelihood_for_sequence = 1.0
            else:
                logger.warning("Chosen plant spots not provided for sophisticated agent")
        else:
            # Naive agents drop crumbs randomly in kitchen on return path
            simulated_door_states = initial_door_states.copy()
            on_return = False
            generated_crumbs = set()
            
            if fridge_access_point is None and agent_type_being_simulated != 'uniform':
                logger.debug("Fridge access point is None, cannot simulate crumb generation")
            
            for coord in sequence:
                # Get node attributes from graph
                if coord in world_state.world_graph.node_to_vid:
                    vid = world_state.world_graph.node_to_vid[coord]
                    node_attrs = world_state.world_graph.igraph.vs[vid]
                    is_kitchen = node_attrs['room'] == 'Kitchen'
                    is_door = node_attrs['is_door']
                else:
                    logger.warning(f"Coordinate {coord} not found in graph during visual evidence calculation")
                    is_kitchen = False
                    is_door = False

                # Check if agent reached fridge (start of return journey)
                if fridge_access_point and coord == fridge_access_point:
                    on_return = True
                
                # Update door states
                if is_door and coord in simulated_door_states:
                    simulated_door_states[coord] = 'open'

                # Drop crumbs in kitchen on return path
                if on_return and is_kitchen and not is_door:
                    generated_crumbs.add(coord)
            
            if crumb_coord_tuple in generated_crumbs:
                likelihood_for_sequence = 1.0 / current_middle_len
        
        total_likelihood += likelihood_for_sequence
    
    return total_likelihood / num_sequences


# Audio functions

def get_audio_tokens_for_path(world_state: 'World', path_coords: list) -> list:
    """Generate raw audio tokens for an agent traversing a path"""
    raw_tokens = []
    if not path_coords:
        return raw_tokens

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
    Compress sequences of raw tokens into [num_steps_to, fridge_events, num_steps_from] format
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
    """Convert a world coordinate path to compressed audio token sequence"""
    raw_tokens = get_audio_tokens_for_path(world_state, path_coords)
    return parse_raw_audio_tokens(raw_tokens)


def single_segment_audio_likelihood(gt_steps: int, path_steps: int, sigma_factor: float = 0.1) -> float:
    """
    Compute likelihood of observing path_steps given expected gt_steps.
    Uses normalized Gaussian PDF for robustness.
    """
    sigma = max(1.0, gt_steps * sigma_factor) if gt_steps > 0 else 1.0
    
    # Calculate likelihood and normalize by maximum possible likelihood
    likelihood = norm.pdf(path_steps, loc=gt_steps, scale=sigma)
    max_likelihood = norm.pdf(gt_steps, loc=gt_steps, scale=sigma)
    
    if max_likelihood == 0:
        return 1.0 if likelihood == 0 and path_steps == gt_steps else 0.0

    return likelihood / max_likelihood


def generate_ground_truth_audio_sequences(world: 'World', params: SimulationParams) -> list:
    """
    Generate ground truth compressed audio sequences for detective predictions.
    Returns list of sequences like [[steps_to, 'fridge_opened', 'snack_picked_up', 'fridge_closed', steps_from], ...]
    """
    logger = logging.getLogger(__name__)
    ground_truths = []
    
    # Get world geometry information
    fridge_access_point = world.get_fridge_access_point()
    start_coords = world.start_coords
    node_to_vid = world.world_graph.node_to_vid
    igraph = world.world_graph.igraph

    # Convert coordinates to vertex IDs
    start_A_vid = node_to_vid[start_coords['A']]
    start_B_vid = node_to_vid[start_coords['B']]
    fridge_vid = node_to_vid[fridge_access_point]

    # Calculate shortest paths to fridge
    def get_shortest_path_length(source_vid, target_vid):
        path_matrix = igraph.shortest_paths(source=source_vid, target=target_vid)
        return path_matrix[0][0] if path_matrix and path_matrix[0] else float('inf')

    path_A_to_fridge = get_shortest_path_length(start_A_vid, fridge_vid)
    path_B_to_fridge = get_shortest_path_length(start_B_vid, fridge_vid)

    if path_A_to_fridge == float('inf') and path_B_to_fridge == float('inf'):
        logger.warning("Neither agent can reach the fridge. No ground truth sequences generated")
        return []

    # Use shorter path for base calculations
    min_steps_to_fridge = min(path_A_to_fridge, path_B_to_fridge)
    closer_agent_vid = start_A_vid if path_A_to_fridge <= path_B_to_fridge else start_B_vid
    min_steps_from_fridge = get_shortest_path_length(fridge_vid, closer_agent_vid)

    # Ensure valid base steps
    base_steps_to = int(min_steps_to_fridge) if min_steps_to_fridge != float('inf') else params.max_steps + 1
    base_steps_from = int(min_steps_from_fridge) if min_steps_from_fridge != float('inf') else params.max_steps + 1
    
    fridge_events = ['fridge_opened', 'snack_picked_up', 'fridge_closed']

    # Generate variations around base paths
    for to_increment in range(0, params.max_steps + 1, params.audio_gt_step_size):
        steps_to = base_steps_to + to_increment
        if steps_to > params.max_steps:
            continue

        for from_increment in range(0, params.max_steps + 1, params.audio_gt_step_size):
            steps_from = base_steps_from + from_increment
            if steps_from > params.max_steps:
                continue
            
            sequence = [steps_to] + fridge_events + [steps_from]
            if sequence not in ground_truths:
                ground_truths.append(sequence)
                
    # Ensure at least the base case is included
    if not ground_truths and base_steps_to <= params.max_steps and base_steps_from <= params.max_steps:
        ground_truths.append([base_steps_to] + fridge_events + [base_steps_from])

    logger.info(f"Generated {len(ground_truths)} ground truth audio sequences")
    return sorted(ground_truths)
