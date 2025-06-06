"""
Evaluation functionality for analyzing simulation results.

Provides functions for extracting experiment parameters, analyzing path lengths,
and evaluating detective predictions.
"""

import pandas as pd
import json
import ast
import os
import glob
import logging
from typing import Dict, List, Any


logger = logging.getLogger(__name__)


def extract_params_from_path(log_dir_path: str) -> Dict[str, Any]:
    """
    Extracts experiment parameters from a log directory path string.
    Example path: ../../results/trial_name/w0.1_ntemp0.05_stemp0.05_steps25_20240101_120000
    """
    params = {
        'w': None, 'n_temp': None, 's_temp': None, 
        'max_steps': None, 'soph_suspect_sigma': None, 
        'soph_detective_sigma': None, 'noisy_planting_sigma': None,
        'door_close_prob': None, 'audio_gt_step_size': None,
        'audio_segment_similarity_sigma': None, 'sample_paths_suspect': None,
        'sample_paths_detective': None, 'seed': None, 'command': None,
        'mismatched': None, 'param_dir_name': None
    }
    
    dir_name = os.path.basename(log_dir_path)
    params['param_dir_name'] = dir_name

    parts = dir_name.split('_')
    for part in parts:
        if part.startswith('w') and 'w0.' not in part: 
            val_str = part[1:]
            if val_str.replace('.', '', 1).isdigit(): 
                params['w'] = float(val_str)
        elif part.startswith('ntemp'):
            val_str = part[len('ntemp'):]
            if val_str.replace('.', '', 1).isdigit():
                params['n_temp'] = float(val_str)
        elif part.startswith('stemp'):
            val_str = part[len('stemp'):]
            if val_str.replace('.', '', 1).isdigit():
                params['s_temp'] = float(val_str)
        elif part.startswith('steps'):
            val_str = part[len('steps'):]
            if val_str.isdigit():
                params['max_steps'] = int(val_str)

    metadata_path = os.path.join(log_dir_path, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract parameters from metadata, prioritizing metadata over path parsing
            metadata_params = metadata.get('parameters', {})
            if not metadata_params:
                # Try direct metadata access for backwards compatibility
                metadata_params = metadata
            
            params['w'] = metadata_params.get('w', params['w'])
            params['n_temp'] = metadata_params.get('n_temp', params['n_temp'])
            params['s_temp'] = metadata_params.get('s_temp', params['s_temp'])
            params['max_steps'] = metadata_params.get('max_steps', params['max_steps'])
            params['soph_suspect_sigma'] = metadata_params.get('soph_suspect_sigma', params['soph_suspect_sigma'])
            params['soph_detective_sigma'] = metadata_params.get('soph_detective_sigma', params['soph_detective_sigma'])
            params['noisy_planting_sigma'] = metadata_params.get('noisy_planting_sigma', params['noisy_planting_sigma'])
            params['door_close_prob'] = metadata_params.get('door_close_prob', params['door_close_prob'])
            params['audio_gt_step_size'] = metadata_params.get('audio_gt_step_size', params['audio_gt_step_size'])
            params['audio_segment_similarity_sigma'] = metadata_params.get('audio_segment_similarity_sigma', params['audio_segment_similarity_sigma'])
            params['sample_paths_suspect'] = metadata_params.get('sample_paths_suspect', params['sample_paths_suspect'])
            params['sample_paths_detective'] = metadata_params.get('sample_paths_detective', params['sample_paths_detective'])
            params['seed'] = metadata_params.get('seed', params['seed'])
            params['command'] = metadata_params.get('command', params['command'])
            params['mismatched'] = metadata_params.get('mismatched', params['mismatched'])
            
        except Exception as e:
            logger.warning(f"Could not parse metadata.json in {log_dir_path} for detailed params: {e}")
            
    return params


def get_evidence_type_from_metadata(base_dir: str, trial_name: str) -> str:
    """Tries to determine evidence type from metadata.json."""
    metadata_path = os.path.join(base_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(os.path.dirname(base_dir), 'metadata.json')
    if not os.path.exists(metadata_path):
         metadata_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'metadata.json')

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('evidence', 'visual').lower()
        except Exception as e:
            logger.warning(f"Could not parse metadata.json for evidence type: {e}")
    
    return 'visual'  # default fallback


def calculate_avg_path_lengths(csv_filepath: str) -> List[Dict[str, Any]]:
    """
    Calculates and returns average path lengths for each agent and agent_type.
    Determines evidence type from metadata.json.
    """
    logger.info(f"Analyzing path lengths from: {csv_filepath}")
    
    df = pd.read_csv(csv_filepath)
    param_log_dir = os.path.dirname(csv_filepath)
    base_filename = os.path.basename(csv_filepath)
    
    trial_name = "unknown_trial"
    generator_agent_type = "unknown_type"
    parts = base_filename.replace("_sampled_paths_", "$").replace(".csv", "").split("$")
    if len(parts) == 2:
        trial_name = parts[0]
        generator_agent_type = parts[1]

    evidence_type = get_evidence_type_from_metadata(param_log_dir, trial_name)
    # Extract experiment parameters from the param_log_dir path
    experiment_params = extract_params_from_path(param_log_dir)
    logger.info(f"Trial: {trial_name}, GenType: {generator_agent_type}, Evidence: {evidence_type}, Params from dir: {experiment_params}")

    results = []

    for col in ['full_sequence_length', 'to_fridge_sequence_length', 'middle_sequence_length']:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in {csv_filepath}. Path length analysis for it will be impacted.")
            df[col] = pd.NA 

    if evidence_type == 'audio' and 'audio_sequence_compressed' in df.columns:
        def get_audio_total_length(audio_seq_str):
            try:
                seq = ast.literal_eval(audio_seq_str)
                if isinstance(seq, list) and len(seq) == 5 and isinstance(seq[0], int) and isinstance(seq[4], int):
                    return seq[0] + seq[4]
            except (ValueError, SyntaxError, TypeError):
                pass
            return pd.NA
        df['audio_total_compressed_length'] = df['audio_sequence_compressed'].apply(get_audio_total_length)
    else:
        df['audio_total_compressed_length'] = pd.NA

    group_by_cols = []
    if 'agent_type' in df.columns: 
        group_by_cols.append('agent_type')
    if 'agent' in df.columns:
        group_by_cols.append('agent')

    grouped = df.groupby(group_by_cols, observed=True) 
    
    summary_cols_map = {
        'avg_full_sequence_length': 'full_sequence_length',
        'avg_to_fridge_sequence_length': 'to_fridge_sequence_length',
        'avg_middle_sequence_length': 'middle_sequence_length',
        'avg_audio_total_compressed_length': 'audio_total_compressed_length'
    }

    for name, group_data in grouped:
        num_samples = len(group_data)
        current_summary = {
            'trial_name': trial_name,
            'generator_agent_type': generator_agent_type,
            'agent_id': None,
            'evidence_type': evidence_type,
            'num_samples': num_samples,
            **experiment_params 
        }
        
        if len(group_by_cols) == 2: 
            current_summary['generator_agent_type'] = name[0] 
            current_summary['agent_id'] = name[1]
        elif group_by_cols[0] == 'agent':
             current_summary['agent_id'] = name
             current_summary['generator_agent_type'] = generator_agent_type
        elif group_by_cols[0] == 'agent_type':
            current_summary['generator_agent_type'] = name 

        print_line = f"  Trial: {trial_name}, Evidence: {evidence_type}, GenType: {current_summary['generator_agent_type']}, Agent: {current_summary['agent_id'] if current_summary['agent_id'] else 'N/A'} ({num_samples} samples):"
        
        for summary_col_name, data_col_name in summary_cols_map.items():
            if data_col_name in group_data.columns:
                avg_val = group_data[data_col_name].mean(skipna=True)
                current_summary[summary_col_name] = round(avg_val, 2) if pd.notna(avg_val) else None
                if pd.notna(avg_val):
                     print_line += f" Avg {data_col_name}: {avg_val:.2f}"
            else:
                current_summary[summary_col_name] = None
        
        results.append(current_summary)
        print(print_line)

    print("-" * 30)
    return results


def analyze_gt_audio_lengths_vs_predictions(json_filepath: str, num_bins: int = 5) -> List[Dict[str, Any]]:
    """
    Analyzes detective predictions by binning ground truth audio sequence lengths
    and showing average slider, L_A, and L_B for each bin, grouped by the detective's simulated agent type.
    """
    logger.info(f"Analyzing audio length vs predictions from: {json_filepath}")
    
    if not os.path.exists(json_filepath):
        logger.error(f"JSON file not found: {json_filepath}")
        return []
    
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Could not parse JSON file {json_filepath}: {e}")
        return []
    
    if not isinstance(data, list) or len(data) == 0:
        logger.warning(f"No data found in {json_filepath}")
        return []
    
    results = []
    
    # Group by agent type being simulated
    agent_types = {}
    for item in data:
        agent_type = item.get('agent_type_being_simulated', 'unknown')
        if agent_type not in agent_types:
            agent_types[agent_type] = []
        agent_types[agent_type].append(item)
    
    for agent_type, type_data in agent_types.items():
        logger.info(f"Analyzing {len(type_data)} predictions for {agent_type} agents")
        
        # Extract ground truth sequence lengths and predictions
        gt_lengths = []
        predictions = []
        for item in type_data:
            gt_seq = item.get('gt_sequence', [])
            if isinstance(gt_seq, list) and len(gt_seq) >= 5:
                # Audio sequences: [to_steps, event1, event2, event3, from_steps]
                total_length = gt_seq[0] + gt_seq[4] if isinstance(gt_seq[0], int) and isinstance(gt_seq[4], int) else 0
                gt_lengths.append(total_length)
                predictions.append(item.get('prediction', 0.5))
        
        if not gt_lengths:
            logger.warning(f"No valid ground truth sequences found for {agent_type}")
            continue
        
        # Create bins
        min_length = min(gt_lengths)
        max_length = max(gt_lengths)
        if min_length == max_length:
            bin_edges = [min_length - 0.5, max_length + 0.5]
        else:
            bin_edges = [min_length + i * (max_length - min_length) / num_bins for i in range(num_bins + 1)]
        
        # Bin the data
        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            bin_indices = [j for j, length in enumerate(gt_lengths) 
                          if bin_start <= length < bin_end or (i == len(bin_edges) - 2 and length == bin_end)]
            
            if bin_indices:
                bin_predictions = [predictions[j] for j in bin_indices]
                avg_prediction = sum(bin_predictions) / len(bin_predictions)
                
                bin_result = {
                    'agent_type': agent_type,
                    'bin_range': f"{bin_start:.1f}-{bin_end:.1f}",
                    'bin_center': (bin_start + bin_end) / 2,
                    'num_samples': len(bin_indices),
                    'avg_prediction': round(avg_prediction, 3),
                    'min_length': min([gt_lengths[j] for j in bin_indices]),
                    'max_length': max([gt_lengths[j] for j in bin_indices])
                }
                results.append(bin_result)
                
                logger.info(f"  Bin {bin_result['bin_range']}: {bin_result['num_samples']} samples, "
                           f"avg prediction: {bin_result['avg_prediction']:.3f}")
    
    return results


def consolidate_evaluation_results(dir_to_search: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Consolidate evaluation results from multiple experiment directories.
    
    Returns:
        Dictionary with 'path_lengths' and 'audio_predictions' keys containing consolidated results
    """
    logger.info(f"Consolidating evaluation results from: {dir_to_search}")
    
    all_path_length_results = []
    all_audio_prediction_results = []
    
    # Find all CSV files for path length analysis
    csv_pattern = os.path.join(dir_to_search, "**/*_sampled_paths_*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    logger.info(f"Found {len(csv_files)} CSV files for path length analysis")
    
    for csv_file in csv_files:
        try:
            results = calculate_avg_path_lengths(csv_file)
            all_path_length_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
    
    # Find all JSON files for audio prediction analysis
    json_pattern = os.path.join(dir_to_search, "**/*_audio_predictions.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    logger.info(f"Found {len(json_files)} JSON files for audio prediction analysis")
    
    for json_file in json_files:
        try:
            results = analyze_gt_audio_lengths_vs_predictions(json_file)
            all_audio_prediction_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    return {
        'path_lengths': all_path_length_results,
        'audio_predictions': all_audio_prediction_results
    } 