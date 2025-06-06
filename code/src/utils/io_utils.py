import os
import json
import pandas as pd
import numpy as np
import logging
import datetime


def get_json_files(trial):
    """Get JSON files for trials from various possible locations."""
    base_dirs_to_try = [
        # From src/utils/ go up two levels to code/, then to trials/
        os.path.join(os.path.dirname(__file__), '..', '..', 'trials', 'suspect', 'json'),
        # From current working directory (code/)
        os.path.join('.', 'trials', 'suspect', 'json'),
        # Alternative relative paths
        'trials/suspect/json',
        '../trials/suspect/json',
        # Absolute path construction from code root
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'trials', 'suspect', 'json')
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
            if os.path.exists(full_path): 
                return [filename]
            else: 
                raise FileNotFoundError(f"Trial file not found: {full_path}")
    except Exception as e: 
        raise IOError(f"Error accessing trial JSON files: {e}")
    

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


def save_sampled_paths_to_csv(sampled_data, trial_name, param_log_dir, agent_type):
    """Saves the sampled paths to a CSV file."""
    logger = logging.getLogger(__name__)
    paths_data_to_save = []

    for agent in ['A', 'B']:
        agent_path_list = sampled_data.get(agent)
        if not agent_path_list:
            logger.warning(f"No data for agent {agent} ({agent_type}) in {trial_name}.")
            continue

        for path_data in agent_path_list:
            paths_data_to_save.append({
                'trial': trial_name,
                'agent': agent,
                'agent_type': agent_type,
                'path_grid': "N/A",  # path_str logic removed for simplicity
                'full_sequence': str(path_data.get('full_sequence', [])),
                'middle_sequence': str(path_data.get('middle_sequence', [])),
                'to_fridge_sequence': str(path_data.get('to_fridge_sequence', [])),
                'chosen_plant_spot': str(path_data.get('chosen_plant_spot')) if agent_type == 'sophisticated' else "N/A",
                'audio_sequence_compressed': "N/A", # audio_seq_str logic removed
                'full_sequence_length': path_data.get('full_sequence_length', -1),
                'to_fridge_sequence_length': path_data.get('to_fridge_sequence_length', -1),
                'middle_sequence_length': path_data.get('middle_sequence_length', -1)
            })

    df = pd.DataFrame(paths_data_to_save)
    csv_path = os.path.join(param_log_dir, f'{trial_name}_sampled_paths_{agent_type}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} sampled path records to {csv_path}")


def save_grid_to_json(grid, filename):
    """Save grid data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(grid, f)


def ensure_serializable(obj):
    """Ensure object is JSON serializable."""
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
    return obj 
