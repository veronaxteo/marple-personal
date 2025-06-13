"""
I/O utilities for data handling, including loading/saving JSON, CSV, and caching.
"""

import os
import json
import pandas as pd
import logging
from typing import Dict
import numpy as np
from src.cfg import SimulationConfig
import datetime

logger = logging.getLogger(__name__)


def load_json(file_path: str) -> Dict:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        return {}

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
    

def create_param_dir(log_dir, trial_name, evidence_type, max_steps=25, model_type="rsm", cost_weight=None, naive_temp=None, soph_temp=None):
    """Creates parameter-specific log directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_type == "rsm":
        param_subdir = f'w{cost_weight}_{evidence_type}_ntemp{naive_temp}_stemp{soph_temp}_steps{max_steps}_{timestamp}'
    elif model_type == "uniform":
        param_subdir = f'uniform_{evidence_type}_steps{max_steps}_{timestamp}'
    elif model_type == "empirical":
        param_subdir = 'empirical'
    else:
        param_subdir = 'unknown_model'

    param_log_dir = os.path.join(log_dir, trial_name, param_subdir)
    os.makedirs(param_log_dir, exist_ok=True)
    return param_log_dir


def save_json(data: Dict, file_path: str, indent: int = 4):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_world_config(world_name: str) -> Dict:
    """Load world configuration from a JSON file."""
    file_path = os.path.join('worlds', f"{world_name}.json")
    return load_json(file_path)


def load_human_data(trial_name: str) -> Dict:
    """Load human data for a specific trial."""
    # Assuming human data is stored in a structured way, e.g., data/human/{trial_name}.json
    file_path = os.path.join('data', 'human', f"{trial_name}.json")
    return load_json(file_path)


def save_simulation_config(config: SimulationConfig, log_dir: str):
    """Save simulation configuration to a JSON file in the log directory."""
    metadata = {
        'simulation_config': config.dict()
    }
    file_path = os.path.join(log_dir, 'metadata.json')
    save_json(metadata, file_path)


def save_sampled_paths_to_csv(sampled_paths_by_agent: Dict, trial_name: str, log_dir: str, agent_level: str):
    """Save sampled path data from multiple agents to a single CSV file."""
    if not sampled_paths_by_agent:
        logger.warning(f"No sampled paths to save for {trial_name} ({agent_level})")
        return
        
    log_path = os.path.join(log_dir, f"{trial_name}_sampled_paths_{agent_level}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    all_rows = []
    
    for agent_id, sampled_paths in sampled_paths_by_agent.items():
        if not isinstance(sampled_paths, dict):
            logger.warning(f"Unexpected data format for agent {agent_id} in save_sampled_paths_to_csv. Skipping.")
            continue

        full_sequences = sampled_paths.get('full_sequences', [])
        to_fridge_sequences = sampled_paths.get('to_fridge_sequences', [])
        return_sequences = sampled_paths.get('return_sequences', [])
        chosen_plant_spots = sampled_paths.get('chosen_plant_spots', [])
        audio_sequences = sampled_paths.get('audio_sequences', [])
        full_sequence_lengths = sampled_paths.get('full_sequence_lengths', [])
        to_fridge_sequence_lengths = sampled_paths.get('to_fridge_sequence_lengths', [])
        return_sequence_lengths = sampled_paths.get('return_sequence_lengths', [])

        num_paths = len(full_sequences)

        for i in range(num_paths):
            row = {
                'trial_name': trial_name,
                'agent_level': agent_level,
                'agent_id': agent_id,
                'full_sequence': str(full_sequences[i]) if i < len(full_sequences) else '',
                'to_fridge_sequence': str(to_fridge_sequences[i]) if i < len(to_fridge_sequences) else '',
                'return_sequence': str(return_sequences[i]) if i < len(return_sequences) else '',
                'chosen_plant_spot': str(chosen_plant_spots[i]) if i < len(chosen_plant_spots) else None,
                'audio_sequence': str(audio_sequences[i]) if i < len(audio_sequences) else None,
                'full_sequence_length': full_sequence_lengths[i] if i < len(full_sequence_lengths) else 0,
                'to_fridge_sequence_length': to_fridge_sequence_lengths[i] if i < len(to_fridge_sequence_lengths) else 0,
                'return_sequence_length': return_sequence_lengths[i] if i < len(return_sequence_lengths) else 0,
            }
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(log_path, index=False)
        logger.info(f"Saved {len(all_rows)} sampled paths to {log_path}")
    else:
        logger.warning(f"No data rows were generated to save for {trial_name} ({agent_level})")


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
