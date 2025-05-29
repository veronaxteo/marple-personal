import os
import pandas as pd
import numpy as np
import logging
import pickle
import datetime
import json
from params import SimulationParams
from scipy.ndimage import gaussian_filter
from numpy.random import rand


def normalized_slider_prediction(value_A, value_B):
    if sum([value_A, value_B]) == 0: return 0
    else: return round(100 * (value_B / (value_A + value_B)) - 50, 0)


def softmax_list_vals(vals, temp):
    return np.exp(np.array(vals) / temp) / np.sum(np.exp(np.array(vals) / temp), axis=0)


def create_param_dir(log_dir, trial_name, w=0, naive_temp=0, soph_temp=0, max_steps=25, model_type="rsm"):
    """Creates parameter-specific log directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == "rsm":
        param_subdir = f'w{w}_ntemp{naive_temp}_stemp{soph_temp}_steps{max_steps}_{timestamp}'
    elif model_type == "uniform":
        param_subdir = f'uniform_steps{max_steps}'
    elif model_type == "empirical":
         param_subdir = 'empirical'
    else:
        param_subdir = 'unknown_model'

    param_log_dir = os.path.join(log_dir, trial_name, param_subdir)
    os.makedirs(param_log_dir, exist_ok=True)
    return param_log_dir


def get_json_files(trial):
    # Attempt more robust path finding for trial JSONs
    base_dirs_to_try = [
        os.path.join(os.path.dirname(__file__), '..', 'trials', 'suspect', 'json'), # Assuming utils.py is in src/
        os.path.join('.', 'trials', 'suspect', 'json'), # Relative to CWD
        '../trials/suspect/json' # Original path
    ]
    trial_json_path = None
    for base_dir in base_dirs_to_try:
        candidate_path = os.path.abspath(base_dir)
        if os.path.isdir(candidate_path):
            trial_json_path = candidate_path
            break
    
    if not trial_json_path:
        raise FileNotFoundError(f"Trial JSON directory not found in checked locations: {base_dirs_to_try}")

    try:
        if trial == 'all':
            return sorted([f for f in os.listdir(trial_json_path) if f.endswith('_A1.json')])
        else:
            filename = f'{trial}_A1.json'
            full_path = os.path.join(trial_json_path, filename)
            if os.path.exists(full_path): return [filename]
            else: raise FileNotFoundError(f"Trial file not found: {full_path}")
    except Exception as e: raise IOError(f"Error accessing trial JSON files: {e}")


def get_shortest_paths(igraph_instance, source_vid: int, target_vid: int, vid_to_node_map: dict):
    """Finds all shortest paths between source and target in an igraph Graph."""
    logger = logging.getLogger(__name__)
    try:
        # Returns lists of VIDs
        vid_paths = igraph_instance.get_all_shortest_paths(source_vid, to=target_vid, weights=None, mode='all')
        coord_paths = []
        for vid_path in vid_paths:
            coord_path = [vid_to_node_map[vid] for vid in vid_path]
            coord_paths.append(coord_path)
        return coord_paths
    except Exception as e: # igraph might raise different errors for no path, e.g., IGraphError or if VIDs are invalid
        logger.warning(f"No path found between VIDs {source_vid} and {target_vid} using igraph, or error: {e}")
        return []

# def get_shortest_paths(graph, source, target):
#     """Finds all shortest paths between source and target in graph."""
#     try:
#         return list(nx.all_shortest_paths(graph, source, target))
#     except nx.NetworkXNoPath:
#         logger = logging.getLogger(__name__)
#         logger.warning(f"No path found between {source} and {target}")
#         return []


def get_simple_paths(igraph_instance, source_vid: int, target_vid: int, cutoff: int, vid_to_node_map: dict):
    """Finds all simple paths up to cutoff length using igraph."""
    logger = logging.getLogger(__name__)
    try:
        vid_paths = igraph_instance.get_all_simple_paths(source_vid, to=target_vid, cutoff=cutoff, mode='all')
        coord_paths = []
        for vid_path in vid_paths:
            coord_path = [vid_to_node_map[vid] for vid in vid_path]
            coord_paths.append(coord_path)
        return coord_paths
    except Exception as e:
        logger.warning(f"Error finding simple paths between VIDs {source_vid} and {target_vid} with cutoff {cutoff} using igraph: {e}")
        return []

# def get_simple_paths(graph, source, target, simple_path_cutoff):
#     """Finds all simple paths up to cutoff length."""
#     try:
#         return list(nx.all_simple_paths(graph, source, target, cutoff=simple_path_cutoff))
#     except nx.NetworkXNoPath:
#         logger = logging.getLogger(__name__)
#         logger.warning(f"No path found between {source} and {target}")
#         return []


def save_sampled_paths_to_csv(sampled_data, trial_name, param_log_dir, agent_type):
    """Saves the sampled paths (numbered 2D arrays) to a CSV file."""
    logger = logging.getLogger(__name__)
    paths_data_to_save = []

    for agent in ['A', 'B']:
        agent_data = sampled_data.get(agent)
        if not agent_data:
            logger.warning(f"No data for agent {agent} ({agent_type}) in {trial_name}.")
            continue

        full_sequences = agent_data.get('full_sequences', [])
        middle_sequences = agent_data.get('middle_sequences', [])
        chosen_plant_spots = agent_data.get('chosen_plant_spots', [])
        numbered_arrays = agent_data.get('numbered_arrays', []) # May be empty for audio
        audio_sequences = agent_data.get('audio_sequences', []) # New, for audio
        # Add new keys for path to fridge and path lengths
        to_fridge_sequences = agent_data.get('to_fridge_sequences', [])
        full_sequence_lengths = agent_data.get('full_sequence_lengths', [])
        to_fridge_sequence_lengths = agent_data.get('to_fridge_sequence_lengths', [])
        middle_sequence_lengths = agent_data.get('middle_sequence_lengths', [])

        # Determine the primary list to iterate over for saving records
        # If numbered_arrays exist (visual), use that length. Otherwise, use full_sequences (audio or general).
        num_records = len(numbered_arrays) if numbered_arrays else len(full_sequences)
        if num_records == 0:
            logger.warning(f"No sequences or numbered_arrays to save for agent {agent} ({agent_type}) in {trial_name}.")
            continue

        for i in range(num_records):
            path_str = "N/A"
            if i < len(numbered_arrays) and numbered_arrays[i] is not None:
                path_str = '\n'.join([' '.join(map(str, row)) for row in numbered_arrays[i]])
            
            plant_spot_str = "N/A"
            # chosen_plant_spots are relevant mainly for sophisticated visual
            if agent_type == 'sophisticated' and i < len(chosen_plant_spots) and chosen_plant_spots[i] is not None:
                plant_spot_str = str(chosen_plant_spots[i])
            
            audio_seq_str = "N/A"
            if i < len(audio_sequences) and audio_sequences[i] is not None:
                audio_seq_str = str(audio_sequences[i])

            paths_data_to_save.append({
                'trial': trial_name,
                'agent': agent,
                'agent_type': agent_type, # This is the type of suspect path (e.g. naive_suspect_paths)
                'path_grid': path_str, # Renamed from 'path' for clarity
                'full_sequence_world_coords': str(full_sequences[i]) if i < len(full_sequences) else "N/A",
                'middle_sequence_world_coords': str(middle_sequences[i]) if i < len(middle_sequences) else "N/A",
                'to_fridge_sequence_world_coords': str(to_fridge_sequences[i]) if i < len(to_fridge_sequences) else "N/A",
                'chosen_plant_spot_world_coords': plant_spot_str,
                'audio_sequence_compressed': audio_seq_str, # Added audio sequence
                'full_sequence_length': full_sequence_lengths[i] if i < len(full_sequence_lengths) else -1,
                'to_fridge_sequence_length': to_fridge_sequence_lengths[i] if i < len(to_fridge_sequence_lengths) else -1,
                'middle_sequence_length': middle_sequence_lengths[i] if i < len(middle_sequence_lengths) else -1
            })

    if not paths_data_to_save:
        logger.warning(f"No data generated to save for {trial_name} ({agent_type}).")
        return

    df = pd.DataFrame(paths_data_to_save)
    # The agent_type in the filename refers to the type of suspect paths being saved (e.g., naive_suspect_paths)
    csv_path = os.path.join(param_log_dir, f'{trial_name}_sampled_paths_{agent_type}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} sampled path records to {csv_path}")


def smooth_likelihood_grid(raw_likelihoods_map: dict, world, sigma: float) -> dict:
    """
    Applies Gaussian smoothing to a grid of likelihoods.

    Returns:
        dict: A dictionary mapping world coordinate tuples to smoothed likelihood values.
    """
    if not sigma > 0:
        return raw_likelihoods_map

    kitchen_width = world.coordinate_mapper.kitchen_width
    kitchen_height = world.coordinate_mapper.kitchen_height
    grid_raw_likelihoods = np.zeros((kitchen_height, kitchen_width), dtype=float)

    for wc_tuple, likelihood in raw_likelihoods_map.items():
        kc = world.world_to_kitchen_coords(wc_tuple[0], wc_tuple[1])
        if kc: 
            grid_raw_likelihoods[kc[1], kc[0]] = likelihood
    
    grid_smoothed_likelihoods = gaussian_filter(grid_raw_likelihoods, sigma=sigma)
    
    smoothed_likelihoods_map = {}

    possible_crumb_coords = world.get_valid_kitchen_crumb_coords_world()
    if not possible_crumb_coords:
        possible_crumb_coords = raw_likelihoods_map.keys()

    for wc_tuple in possible_crumb_coords:
        kc = world.world_to_kitchen_coords(wc_tuple[0], wc_tuple[1])
        if kc:
            smoothed_likelihoods_map[wc_tuple] = grid_smoothed_likelihoods[kc[1], kc[0]]
            
    return smoothed_likelihoods_map


furniture_size = {
    'bed': [3, 2],
    'sofa': [3, 2],
    'light': [1, 2],
    'table': [3, 2],
    'side_table': [1, 1],
    'electric_refrigerator': [2, 3],
    'tv': [2, 2]
}

def flip(p):
    if rand() < p:
        return True
    return False


def ensure_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_serializable(i) for i in obj)
    # Add other non-serializable types here if necessary
    return obj


def save_grid_to_json(grid, filename):
    with open(filename, 'w') as f:
        json.dump(grid, f)


def load_simple_path_sequences(log_dir_base: str, trial_name: str, w_t0: 'World', params: SimulationParams, max_steps_visual_middle: int):
    logger = logging.getLogger(__name__)
    w_t0_hash = hash(str(w_t0.info))
    evidence_type_str = params.evidence
    
    # Determine a relevant step cutoff for the filename, could be different per modality
    # For simplicity, we can use a general max_steps from params if specific ones are too complex for filename, 
    # or make filename more complex. Using max_steps_visual_middle for now as a proxy for complexity.
    # Or, more robustly, include all relevant step parameters if they vary widely.
    # Let's make the cache specific to the major step parameters used in path generation for that modality.
    if evidence_type_str == 'audio':
        # Audio path generation now depends on params.max_steps
        cache_key_steps = f"a_steps{params.max_steps}"
    else: # visual or other
        cache_key_steps = f"v_mid{max_steps_visual_middle}" # Keep visual distinct for now

    pickle_path = os.path.join(log_dir_base, trial_name, f'simple_paths_{evidence_type_str}_{w_t0_hash}_{cache_key_steps}.pkl')
    
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, dict) and 'A' in cached_data and 'B' in cached_data:
                    paths_A_tuple = cached_data['A']
                    paths_B_tuple = cached_data['B']
                    # Basic check for 4-tuple structure
                    if isinstance(paths_A_tuple, tuple) and len(paths_A_tuple) == 4 and \
                       isinstance(paths_B_tuple, tuple) and len(paths_B_tuple) == 4:
                        logger.info(f"Loaded cached {evidence_type_str} simple path sequences for {trial_name} from {pickle_path}.")
                        # Log counts for diagnosis
                        if evidence_type_str == 'visual':
                            logger.info(f"  A: P1:{len(paths_A_tuple[0])}, P2:{len(paths_A_tuple[1])}, P3:{len(paths_A_tuple[2])}")
                            logger.info(f"  B: P1:{len(paths_B_tuple[0])}, P2:{len(paths_B_tuple[1])}, P3:{len(paths_B_tuple[2])}")
                        elif evidence_type_str == 'audio':
                            logger.info(f"  A: P1(S->F):{len(paths_A_tuple[0])}, P_FS(F->S):{len(paths_A_tuple[3])}")
                            logger.info(f"  B: P1(S->F):{len(paths_B_tuple[0])}, P_FS(F->S):{len(paths_B_tuple[3])}")
                        return paths_A_tuple, paths_B_tuple
                    else:
                        logger.warning(f"Cached {evidence_type_str} path tuples have incorrect structure for {trial_name}. Recomputing.")
                else:
                    logger.warning(f"Cached {evidence_type_str} path data is not in expected dict format for {trial_name}. Recomputing.")                    
        except Exception as e:
            logger.warning(f"Error loading cached {evidence_type_str} paths for {trial_name}: {e}. Recomputing.")

    logger.info(f"Computing {evidence_type_str} simple path sequences for {trial_name} (A and B)... Filename: {pickle_path}")
    
    simple_paths_A_p1, simple_paths_A_p2, simple_paths_A_p3, simple_paths_A_fs = w_t0.get_subgoal_simple_path_sequences(
        agent_id='A', params=params, evidence_type=evidence_type_str, max_steps_middle=max_steps_visual_middle
    )
    simple_paths_B_p1, simple_paths_B_p2, simple_paths_B_p3, simple_paths_B_fs = w_t0.get_subgoal_simple_path_sequences(
        agent_id='B', params=params, evidence_type=evidence_type_str, max_steps_middle=max_steps_visual_middle
    )