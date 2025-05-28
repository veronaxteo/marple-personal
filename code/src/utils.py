import os
import pandas as pd
import numpy as np
import logging
import pickle
import random
import networkx as nx
import datetime

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
    trial_json_path = '../trials/suspect/json'
    if not os.path.isdir(trial_json_path):
        raise FileNotFoundError(f"Trial JSON directory not found: {trial_json_path}")
    try:
        if trial == 'all':
            return sorted([f for f in os.listdir(trial_json_path) if f.endswith('A1.json')])
        else:
            filename = f'{trial}_A1.json'
            full_path = os.path.join(trial_json_path, filename)
            if os.path.exists(full_path): return [filename]
            else: raise FileNotFoundError(f"Trial file not found: {full_path}")
    except Exception as e: raise IOError(f"Error accessing trial JSON files: {e}")


def get_shortest_paths(graph, source, target):
    """Finds all shortest paths between source and target in graph."""
    try:
        return list(nx.all_shortest_paths(graph, source, target))
    except nx.NetworkXNoPath:
        logger = logging.getLogger(__name__)
        logger.warning(f"No path found between {source} and {target}")
        return []


def get_simple_paths(graph, source, target, simple_path_cutoff):
    """Finds all simple paths up to cutoff length."""
    try:
        return list(nx.all_simple_paths(graph, source, target, cutoff=simple_path_cutoff))
    except nx.NetworkXNoPath:
        logger = logging.getLogger(__name__)
        logger.warning(f"No path found between {source} and {target}")
        return []
    

# TODO: clean up (maybe separate into two functions, one for loading if simple paths exist and one for computing them)
def load_simple_path_sequences(log_dir_base, trial_name, world, max_steps):
    """
    Loads simple path sequences from .pkl files if they exist.
    If not found, computes them using world.get_subgoal_simple_path_sequences, saves them, and then returns them.
    """
    logger = logging.getLogger(__name__)
    paths_dir = os.path.join(log_dir_base, 'simple_paths')
    os.makedirs(paths_dir, exist_ok=True)

    path_file_A = os.path.join(paths_dir, f'{trial_name}_simple_paths_A.pkl')
    path_file_B = os.path.join(paths_dir, f'{trial_name}_simple_paths_B.pkl')

    sequences_A, sequences_B = None, None

    # Try loading cached simple paths
    if os.path.exists(path_file_A) and os.path.exists(path_file_B):
        try:
            with open(path_file_A, 'rb') as f: sequences_A = pickle.load(f)
            with open(path_file_B, 'rb') as f: sequences_B = pickle.load(f)
            logger.info(f"Loaded cached simple path sequences for {trial_name}.")

        except Exception as e:
            logger.warning(f"Error loading cached sequences for {trial_name}: {e}. Recomputing.")
            sequences_A, sequences_B = None, None 

    # If simple path loading failed or files don't exist, compute simple paths
    if sequences_A is None or sequences_B is None:
        logger.info(f"Computing simple path sequences for {trial_name}...")
        try:
            logger.info("Computing paths for Agent A...")
            sequences_A = world.get_subgoal_simple_path_sequences('A', max_steps)
            logger.info("Computing paths for Agent B...")
            sequences_B = world.get_subgoal_simple_path_sequences('B', max_steps)

            try:
                with open(path_file_A, 'wb') as f: pickle.dump(sequences_A, f)
                with open(path_file_B, 'wb') as f: pickle.dump(sequences_B, f)
                logger.info(f"Saved computed sequences to {paths_dir}")
            except Exception as e:
                logger.error(f"Error saving computed sequences for {trial_name}: {e}")

        except Exception as e:
            logger.error(f"Fatal error computing simple path sequences for {trial_name}: {e}")
            return None, None 

    if not (isinstance(sequences_A, list) and len(sequences_A) == 3 and
            isinstance(sequences_B, list) and len(sequences_B) == 3):
        logger.error(f"Computed sequence data for {trial_name} has incorrect structure.")
        return None, None

    return sequences_A, sequences_B


def save_sampled_paths_to_csv(sampled_data, trial_name, param_log_dir, agent_type):
    """Saves the sampled paths (numbered 2D arrays) to a CSV file."""
    logger = logging.getLogger(__name__)
    paths_data_to_save = []

    for agent in ['A', 'B']:
        agent_data = sampled_data.get(agent)
        if not agent_data or 'numbered_arrays' not in agent_data:
            logger.warning(f"No 'numbered_arrays' data for agent {agent} ({agent_type}) in {trial_name}.")
            continue

        numbered_arrays = agent_data['numbered_arrays']
        full_sequences = agent_data.get('full_sequences', [])
        middle_sequences = agent_data.get('middle_sequences', [])
        chosen_plant_spots = agent_data.get('chosen_plant_spots', [])

        for i, grid in enumerate(numbered_arrays):
            path_str = '\n'.join([' '.join(map(str, row)) for row in grid])
            plant_spot_str = "N/A"
            if agent_type == 'sophisticated' and i < len(chosen_plant_spots) and chosen_plant_spots[i] is not None:
                plant_spot_str = str(chosen_plant_spots[i])
            
            paths_data_to_save.append({
                'trial': trial_name,
                'agent': agent,
                'agent_type': agent_type,
                'path': path_str, 
                'full_sequence_world_coords': str(full_sequences[i]) if i < len(full_sequences) else "N/A",
                'middle_sequence_world_coords': str(middle_sequences[i]) if i < len(middle_sequences) else "N/A",
                'chosen_plant_spot_world_coords': plant_spot_str
            })

    if not paths_data_to_save:
        logger.warning(f"No numbered array data generated to save for {trial_name} ({agent_type}).")
        return

    df = pd.DataFrame(paths_data_to_save)
    csv_path = os.path.join(param_log_dir, f'{trial_name}_sampled_paths_{agent_type}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} numbered path arrays to {csv_path}")


def smooth_likelihood_grid(raw_likelihoods_map: dict, world, sigma: float) -> dict:
    """
    Applies Gaussian smoothing to a grid of likelihoods.

    Returns:
        dict: A dictionary mapping world coordinate tuples to smoothed likelihood values.
    """
    if not sigma > 0:
        return raw_likelihoods_map

    kitchen_width = world.kitchen_width
    kitchen_height = world.kitchen_height
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