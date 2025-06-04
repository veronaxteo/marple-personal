import os
import json
import pandas as pd
import numpy as np
import logging
import datetime


def get_json_files(trial):
    """Get JSON files for trials from various possible locations."""
    base_dirs_to_try = [
        os.path.join(os.path.dirname(__file__), '..', 'trials', 'suspect', 'json'),
        os.path.join('.', 'trials', 'suspect', 'json'),
        '../trials/suspect/json'
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
    

def create_param_dir(log_dir, trial_name, cfg, model_type="rsm"):
    """Creates parameter-specific log directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == "rsm":
        param_subdir = f'w{cfg.sampling.cost_weight}_ntemp{cfg.sampling.naive_temp}_stemp{cfg.sampling.sophisticated_temp}_steps{cfg.sampling.max_steps}_{timestamp}'
    elif model_type == "uniform":
        param_subdir = f'uniform_steps{cfg.sampling.max_steps}'
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
        agent_data = sampled_data.get(agent)
        if not agent_data:
            logger.warning(f"No data for agent {agent} ({agent_type}) in {trial_name}.")
            continue

        full_sequences = agent_data.get('full_sequences', [])
        middle_sequences = agent_data.get('middle_sequences', [])
        chosen_plant_spots = agent_data.get('chosen_plant_spots', [])
        numbered_arrays = agent_data.get('numbered_arrays', [])
        audio_sequences = agent_data.get('audio_sequences', [])
        to_fridge_sequences = agent_data.get('to_fridge_sequences', [])
        full_sequence_lengths = agent_data.get('full_sequence_lengths', [])
        to_fridge_sequence_lengths = agent_data.get('to_fridge_sequence_lengths', [])
        middle_sequence_lengths = agent_data.get('middle_sequence_lengths', [])

        num_records = len(numbered_arrays) if numbered_arrays else len(full_sequences)
        if num_records == 0:
            logger.warning(f"No sequences or numbered_arrays to save for agent {agent} ({agent_type}) in {trial_name}.")
            continue

        for i in range(num_records):
            path_str = "N/A"
            if i < len(numbered_arrays) and numbered_arrays[i] is not None:
                path_str = '\n'.join([' '.join(map(str, row)) for row in numbered_arrays[i]])
            
            plant_spot_str = "N/A"
            if agent_type == 'sophisticated' and i < len(chosen_plant_spots) and chosen_plant_spots[i] is not None:
                plant_spot_str = str(chosen_plant_spots[i])
            
            audio_seq_str = "N/A"
            if i < len(audio_sequences) and audio_sequences[i] is not None:
                audio_seq_str = str(audio_sequences[i])

            paths_data_to_save.append({
                'trial': trial_name,
                'agent': agent,
                'agent_type': agent_type,
                'path_grid': path_str,
                'full_sequence_world_coords': str(full_sequences[i]) if i < len(full_sequences) else "N/A",
                'middle_sequence_world_coords': str(middle_sequences[i]) if i < len(middle_sequences) else "N/A",
                'to_fridge_sequence_world_coords': str(to_fridge_sequences[i]) if i < len(to_fridge_sequences) else "N/A",
                'chosen_plant_spot_world_coords': plant_spot_str,
                'audio_sequence_compressed': audio_seq_str,
                'full_sequence_length': full_sequence_lengths[i] if i < len(full_sequence_lengths) else -1,
                'to_fridge_sequence_length': to_fridge_sequence_lengths[i] if i < len(to_fridge_sequence_lengths) else -1,
                'middle_sequence_length': middle_sequence_lengths[i] if i < len(middle_sequence_lengths) else -1
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
