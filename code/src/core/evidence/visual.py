"""
Visual evidence processing functionality.

Handles visual evidence calculations including crumb dropping simulation
for both naive (random) and sophisticated (strategic) agents.
"""

import logging
from typing import List, Tuple, Optional


def get_visual_evidence_likelihood(crumb_coord_tuple: Tuple[int, int], 
                                 agent_full_sequences: List,
                                 agent_middle_sequences: List,
                                 world_state: 'World', 
                                 agent_type_being_simulated: str = 'naive',
                                 chosen_plant_spots_for_sequences: Optional[List] = None) -> float:
    """
    Calculate likelihood of observing a crumb at given coordinates.
    Handles both naive (random crumb dropping) and sophisticated (strategic planting) agents.
    
    Args:
        crumb_coord_tuple: (x, y) coordinates where crumb was observed
        agent_full_sequences: List of complete agent path sequences
        agent_middle_sequences: List of middle path segments (fridge to door)
        world_state: World object containing environment information
        agent_type_being_simulated: 'naive' or 'sophisticated'
        chosen_plant_spots_for_sequences: List of deliberately chosen plant spots for sophisticated agents
        
    Returns:
        Float likelihood value between 0 and 1
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
                    pass  # No match, keep likelihood_for_sequence = 0.0
            else:
                logger.warning(f"Chosen plant spots not provided for sophisticated agent: spots={chosen_plant_spots_for_sequences is not None}, len_check={len(chosen_plant_spots_for_sequences) if chosen_plant_spots_for_sequences else 'N/A'}, num_seq={num_sequences}")
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
    
    final_likelihood = total_likelihood / num_sequences
    
    return final_likelihood 
