import pandas as pd
import json
import ast
import os
import glob
import argparse
import logging
import sys

logger = logging.getLogger(__name__)

def extract_cfg_from_path(log_dir_path: str) -> dict:
    """
    Extracts experiment parameters from a log directory path string.
    Example path: ../../results/trial_name/w0.1_ntemp0.05_stemp0.05_steps25_20240101_120000
    """
    cfg = {
        'w': None, 'cfg.sampling.naive_temp': None, 'cfg.sampling.sophisticated_temp': None, 
        'cfg.sampling.max_steps': None, 'soph_suspect_sigma': None, 
        'soph_detective_sigma': None, 'noisy_planting_sigma': None,
        'cfg.door_close_prob': None, 'audio_gt_step_size': None,
        'cfg.evidence.audio_similarity_sigma': None, 'cfg.sampling.num_suspect_paths': None,
        'cfg.sampling.num_suspect_paths': None, 'seed': None, 'command': None,
        'mismatched': None, 'param_dir_name': None
    }
    
    dir_name = os.path.basename(log_dir_path)
    cfg['param_dir_name'] = dir_name

    parts = dir_name.split('_')
    for part in parts:
        if part.startswith('w') and 'w0.' not in part: 
            val_str = part[1:]
            if val_str.replace('.', '', 1).isdigit(): 
                cfg['w'] = float(val_str)
        elif part.startswith('ntemp'):
            val_str = part[len('ntemp'):]
            if val_str.replace('.', '', 1).isdigit():
                cfg['cfg.sampling.naive_temp'] = float(val_str)
        elif part.startswith('stemp'):
            val_str = part[len('stemp'):]
            if val_str.replace('.', '', 1).isdigit():
                cfg['cfg.sampling.sophisticated_temp'] = float(val_str)
        elif part.startswith('steps'):
            val_str = part[len('steps'):]
            if val_str.isdigit():
                cfg['cfg.sampling.max_steps'] = int(val_str)

    metadata_path = os.path.join(log_dir_path, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract parameters from metadata, prioritizing metadata over path parsing
            metadata_cfg = metadata.get('parameters', {})
            if not metadata_cfg:
                # Try direct metadata access for backwards compatibility
                metadata_cfg = metadata
            
            cfg['w'] = metadata_cfg.get('w', cfg['w'])
            cfg['cfg.sampling.naive_temp'] = metadata_cfg.get('cfg.sampling.naive_temp', cfg['cfg.sampling.naive_temp'])
            cfg['cfg.sampling.sophisticated_temp'] = metadata_cfg.get('cfg.sampling.sophisticated_temp', cfg['cfg.sampling.sophisticated_temp'])
            cfg['cfg.sampling.max_steps'] = metadata_cfg.get('cfg.sampling.max_steps', cfg['cfg.sampling.max_steps'])
            cfg['soph_suspect_sigma'] = metadata_cfg.get('soph_suspect_sigma', cfg['soph_suspect_sigma'])
            cfg['soph_detective_sigma'] = metadata_cfg.get('soph_detective_sigma', cfg['soph_detective_sigma'])
            cfg['noisy_planting_sigma'] = metadata_cfg.get('noisy_planting_sigma', cfg['noisy_planting_sigma'])
            cfg['cfg.door_close_prob'] = metadata_cfg.get('cfg.door_close_prob', cfg['cfg.door_close_prob'])
            cfg['audio_gt_step_size'] = metadata_cfg.get('audio_gt_step_size', cfg['audio_gt_step_size'])
            cfg['cfg.evidence.audio_similarity_sigma'] = metadata_cfg.get('cfg.evidence.audio_similarity_sigma', cfg['cfg.evidence.audio_similarity_sigma'])
            cfg['cfg.sampling.num_suspect_paths'] = metadata_cfg.get('cfg.sampling.num_suspect_paths', cfg['cfg.sampling.num_suspect_paths'])
            cfg['cfg.sampling.num_suspect_paths'] = metadata_cfg.get('cfg.sampling.num_suspect_paths', cfg['cfg.sampling.num_suspect_paths'])
            cfg['seed'] = metadata_cfg.get('seed', cfg['seed'])
            cfg['command'] = metadata_cfg.get('command', cfg['command'])
            cfg['mismatched'] = metadata_cfg.get('mismatched', cfg['mismatched'])
            
        except Exception as e:
            logger.warning(f"Could not parse metadata.json in {log_dir_path} for detailed cfg: {e}")
            
    return cfg

def get_cfg.evidence.type_type_from_metadata(base_dir: str, trial_name: str) -> str:
    """Tries to determine cfg.evidence.type type from metadata.json."""
    metadata_path = os.path.join(base_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(os.path.dirname(base_dir), 'metadata.json')
    if not os.path.exists(metadata_path):
         metadata_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'metadata.json')

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('cfg.evidence.type', 'visual').lower()
        except Exception as e:
            logger.warning(f"Could not parse metadata.json for cfg.evidence.type type: {e}")
    
    return 'visual'  # default fallback

def calculate_avg_path_lengths(csv_filepath: str) -> list:
    """
    Calculates and returns average path lengths for each agent and agent_type.
    Determines cfg.evidence.type type from metadata.json.
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

    cfg.evidence.type_type = get_cfg.evidence.type_type_from_metadata(param_log_dir, trial_name)
    # Extract experiment parameters from the param_log_dir path
    experiment_cfg = extract_cfg_from_path(param_log_dir)
    logger.info(f"Trial: {trial_name}, GenType: {generator_agent_type}, cfg.evidence.type: {cfg.evidence.type_type}, cfg from dir: {experiment_cfg}")

    results = []

    for col in ['full_sequence_length', 'to_fridge_sequence_length', 'middle_sequence_length']:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in {csv_filepath}. Path length analysis for it will be impacted.")
            df[col] = pd.NA 

    if cfg.evidence.type_type == 'audio' and 'audio_sequence_compressed' in df.columns:
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
            'cfg.evidence.type_type': cfg.evidence.type_type,
            'num_samples': num_samples,
            **experiment_cfg 
        }
        
        if len(group_by_cols) == 2: 
            current_summary['generator_agent_type'] = name[0] 
            current_summary['agent_id'] = name[1]
        elif group_by_cols[0] == 'agent':
             current_summary['agent_id'] = name
             current_summary['generator_agent_type'] = generator_agent_type
        elif group_by_cols[0] == 'agent_type':
            current_summary['generator_agent_type'] = name 

        print_line = f"  Trial: {trial_name}, cfg.evidence.type: {cfg.evidence.type_type}, GenType: {current_summary['generator_agent_type']}, Agent: {current_summary['agent_id'] if current_summary['agent_id'] else 'N/A'} ({num_samples} samples):"
        
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


def analyze_gt_audio_lengths_vs_predictions(json_filepath: str, num_bins: int = 5):
    """
    Analyzes detective predictions by binning ground truth audio sequence lengths
    and showing average slider, L_A, and L_B for each bin, grouped by the detective's simulated agent type.
    """
    print(f"\nAnalyzing ground truth audio lengths vs. predictions from: {json_filepath}")
    with open(json_filepath, 'r') as f:
        predictions_data = json.load(f)

    processed_entries = []
    for entry in predictions_data:
        # Handle both old and new JSON formats
        if 'ground_truth_audio_sequence' in entry:
            # Old format
            gt_seq = entry.get('ground_truth_audio_sequence')
            slider = entry.get('slider')
            l_a = entry.get('L_A_given_gt')
            l_b = entry.get('L_B_given_gt')
            agent_type_sim = entry.get('agent_type_simulated')
        else:
            # New format
            gt_seq = entry.get('gt_sequence')
            slider = entry.get('prediction')
            l_a = entry.get('likelihood_A')
            l_b = entry.get('likelihood_B')
            # Extract agent type from filename since it's not in the JSON
            base_filename = os.path.basename(json_filepath)
            if '_naive_' in base_filename:
                agent_type_sim = 'naive'
            elif '_sophisticated_' in base_filename:
                agent_type_sim = 'sophisticated'
            elif '_uniform_' in base_filename:
                agent_type_sim = 'uniform'
            else:
                agent_type_sim = 'unknown'
        
        if gt_seq and isinstance(gt_seq, list) and len(gt_seq) == 5 and \
           isinstance(gt_seq[0], int) and isinstance(gt_seq[4], int) and \
           slider is not None and l_a is not None and l_b is not None and agent_type_sim is not None:
            total_length = gt_seq[0] + gt_seq[4]
            processed_entries.append({
                'gt_length': total_length, 
                'slider': slider, 
                'L_A': l_a, 
                'L_B': l_b, 
                'agent_type_simulated': agent_type_sim
            })

    if not processed_entries:
        print("No valid prediction entries with all required fields found.")
        return []

    df = pd.DataFrame(processed_entries)
    results_list = []

    param_log_dir = os.path.dirname(json_filepath) 
    experiment_cfg = extract_cfg_from_path(param_log_dir)

    base_filename = os.path.basename(json_filepath)
    # New format: {trial_name}_{agent_type}_{cfg.evidence.type}_predictions.json
    # Extract trial name from new format
    filename_parts = base_filename.replace("_predictions.json", "").split("_")
    trial_name_from_file = filename_parts[0] if len(filename_parts) > 0 else "unknown_trial"

    print("\nOverall Ground Truth Audio Length vs. Predictions Summary:")
    print(f"  Min GT Length: {df['gt_length'].min()}, Max GT Length: {df['gt_length'].max()}")

    for sim_type, group_df in df.groupby('agent_type_simulated'):
        print(f"\n--- Predictions by Detective Modeling Agent Type: '{sim_type.upper()}' ---")

        if group_df['gt_length'].nunique() <= num_bins / 2 or group_df['gt_length'].nunique() <= 1:
            print("  Not enough distinct GT lengths for detailed binning. Showing overall averages:")
            avg_gt_len = group_df['gt_length'].mean()
            avg_slider = group_df['slider'].mean()
            avg_l_a = group_df['L_A'].mean()
            avg_l_b = group_df['L_B'].mean()
            count = len(group_df)
            print(f"    Overall for '{sim_type}': Avg GT Length: {avg_gt_len:.2f}, Avg Slider: {avg_slider:.2f}, Avg L_A: {avg_l_a:.3f}, Avg L_B: {avg_l_b:.3f} (Count: {count})")
            results_list.append({
                'trial_name': trial_name_from_file, 
                'agent_type_simulated': sim_type, 
                'gt_length_bin': 'Overall',
                'avg_gt_length': round(avg_gt_len, 2) if pd.notna(avg_gt_len) else None,
                'avg_slider': round(avg_slider, 2) if pd.notna(avg_slider) else None,
                'avg_L_A': round(avg_l_a, 3) if pd.notna(avg_l_a) else None,
                'avg_L_B': round(avg_l_b, 3) if pd.notna(avg_l_b) else None,
                'count': count,
                **experiment_cfg # Add extracted experiment parameters
            })
            continue
        
        try:
            if group_df['gt_length'].nunique() > num_bins * 2:
                 try:
                    group_df['length_bin'] = pd.qcut(group_df['gt_length'], q=num_bins, precision=0, duplicates='drop')
                 except ValueError:
                    group_df['length_bin'] = pd.cut(group_df['gt_length'], bins=num_bins, precision=0, include_lowest=True, duplicates='drop')    
            else:
                group_df['length_bin'] = pd.cut(group_df['gt_length'], bins=num_bins, precision=0, include_lowest=True, duplicates='drop')
            
            binned_summary = group_df.groupby('length_bin', observed=False).agg(
                avg_slider=('slider', 'mean'),
                avg_L_A=('L_A', 'mean'),
                avg_L_B=('L_B', 'mean'),
                count=('gt_length', 'count')
            ).reset_index()
            
            print("  Binned GT Audio Length Summary:")
            for _, row in binned_summary.iterrows():
                if row['count'] > 0: # Only print bins with data
                    print(f"    GT Length Bin: {str(row['length_bin']):<20} -> Avg Slider: {row['avg_slider']:6.2f}, Avg L_A: {row['avg_L_A']:.3f}, Avg L_B: {row['avg_L_B']:.3f} (Count: {row['count']})")
                    results_list.append({
                        'trial_name': trial_name_from_file,
                        'agent_type_simulated': sim_type,
                        'gt_length_bin': str(row['length_bin']),
                        'avg_gt_length': None, # Bin is a range, specific avg gt_length per bin not directly stored here but could be if needed
                        'avg_slider': round(row['avg_slider'], 2) if pd.notna(row['avg_slider']) else None,
                        'avg_L_A': round(row['avg_L_A'], 3) if pd.notna(row['avg_L_A']) else None,
                        'avg_L_B': round(row['avg_L_B'], 3) if pd.notna(row['avg_L_B']) else None,
                        'count': int(row['count']),
                        **experiment_cfg # Add extracted experiment parameters
                    })
        except Exception as e:
            print(f"  Error during binning or summarization for '{sim_type}': {e}")
            avg_gt_len = group_df['gt_length'].mean()
            avg_slider = group_df['slider'].mean()
            avg_l_a = group_df['L_A'].mean()
            avg_l_b = group_df['L_B'].mean()
            count = len(group_df)
            print(f"    Fallback Overall for '{sim_type}': Avg GT Length: {avg_gt_len:.2f}, Avg Slider: {avg_slider:.2f}, Avg L_A: {avg_l_a:.3f}, Avg L_B: {avg_l_b:.3f} (Count: {count})")
            results_list.append({
                'trial_name': trial_name_from_file, 
                'agent_type_simulated': sim_type, 
                'gt_length_bin': 'Fallback_Overall',
                'avg_gt_length': round(avg_gt_len, 2) if pd.notna(avg_gt_len) else None,
                'avg_slider': round(avg_slider, 2) if pd.notna(avg_slider) else None,
                'avg_L_A': round(avg_l_a, 3) if pd.notna(avg_l_a) else None,
                'avg_L_B': round(avg_l_b, 3) if pd.notna(avg_l_b) else None,
                'count': count,
                **experiment_cfg # Add extracted experiment parameters
            })

    print("-" * 30)
    return results_list


def main_evaluate(dir_to_search: str):
    """Main function to find and process files for evaluation."""
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', stream=sys.stdout)

    logger.info(f"Starting evaluation in directory: {dir_to_search}")

    # Updated to include uniform agent type from UniformSimulator
    csv_search_pattern_sampled_paths = os.path.join(dir_to_search, '**', '*_sampled_paths_*.csv')

    all_csv_files = set()
    all_csv_files.update(glob.glob(csv_search_pattern_sampled_paths, recursive=True))
    
    all_path_length_results = []
    processed_csv_for_avg_length = set()

    for csv_file in all_csv_files:
        if "sampled_paths" in os.path.basename(csv_file) and csv_file not in processed_csv_for_avg_length:
            logger.info(f"--- Analyzing path lengths for: {csv_file} ---")
            try:
                path_length_data = calculate_avg_path_lengths(csv_file)
                if path_length_data:
                    all_path_length_results.extend(path_length_data)
                processed_csv_for_avg_length.add(csv_file)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file} for path lengths: {e}")
    
    all_prediction_analysis_results = []
    # Updated search pattern for new naming convention: {trial_name}_{agent_type}_{cfg.evidence.type}_predictions.json
    # Also search for uniform predictions that might have different naming
    json_search_pattern_audio = os.path.join(dir_to_search, '**', '*_audio_predictions.json')
    json_search_pattern_visual = os.path.join(dir_to_search, '**', '*_visual_predictions.json')
    json_search_pattern_uniform = os.path.join(dir_to_search, '**', '*_uniform_predictions.json')
    
    json_files = glob.glob(json_search_pattern_audio, recursive=True)
    json_files.extend(glob.glob(json_search_pattern_visual, recursive=True))
    json_files.extend(glob.glob(json_search_pattern_uniform, recursive=True))

    if not json_files:
        logger.info("No JSON files matching '*_{cfg.evidence.type}_predictions.json' or '*_uniform_predictions.json' found for detective prediction analysis.")

    for json_file in json_files:
        # Only analyze audio predictions for now, as the function is specific to audio
        if "_audio_predictions.json" in json_file:
            logger.info(f"--- Analyzing GT audio lengths vs. predictions for: {json_file} ---")
            try:
                prediction_analysis_data = analyze_gt_audio_lengths_vs_predictions(json_file)
                if prediction_analysis_data:
                    all_prediction_analysis_results.extend(prediction_analysis_data)
            except Exception as e:
                logger.error(f"Error processing {json_file} for GT length vs. predictions: {e}")

    if all_path_length_results:
        path_lengths_df = pd.DataFrame(all_path_length_results)

        path_lengths_sort_cols = ['trial_name', 'cfg.evidence.type_type', 'generator_agent_type', 'agent_id']
        path_lengths_sort_cols = [col for col in path_lengths_sort_cols if col in path_lengths_df.columns]
        
        if path_lengths_sort_cols:
            path_lengths_df = path_lengths_df.sort_values(by=path_lengths_sort_cols).reset_index(drop=True)
        summary_path_lengths_csv = os.path.join(dir_to_search, "summary_path_lengths.csv")
        path_lengths_df.to_csv(summary_path_lengths_csv, index=False)
        logger.info(f"Saved aggregated path length summary to {summary_path_lengths_csv}")
    else:
        logger.info("No path length results to save.")

    if all_prediction_analysis_results:
        predictions_df = pd.DataFrame(all_prediction_analysis_results)
        predictions_sort_cols = ['trial_name', 'agent_type_simulated', 'gt_length_bin']
        predictions_sort_cols = [col for col in predictions_sort_cols if col in predictions_df.columns]
        if predictions_sort_cols:
            predictions_df = predictions_df.sort_values(by=predictions_sort_cols).reset_index(drop=True)
        summary_predictions_csv = os.path.join(dir_to_search, "summary_detective_predictions.csv")
        predictions_df.to_csv(summary_predictions_csv, index=False)
        logger.info(f"Saved aggregated detective predictions summary to {summary_predictions_csv}")
    else:
        logger.info("No detective prediction results to save.")

    logger.info("Evaluation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate simulation results by analyzing audio sequence lengths and detective predictions.")
    parser.add_argument("--dir", type=str, required=True, help="Root directory to search recursively for result files (CSV and JSON).")
    
    args = parser.parse_args()
    main_evaluate(args.dir)
