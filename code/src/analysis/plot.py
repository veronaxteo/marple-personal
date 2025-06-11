"""
Core plotting functionality for simulation visualization.

Provides functions for creating heatmaps, smoothing comparison plots, and path visualizations.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
import json
import logging
from typing import Dict
from ..core.world import World
from ..utils.math_utils import smooth_likelihoods_old,compute_all_graph_neighbors, smooth_likelihoods


def plot_smoothing_comparison(trial_name: str, param_log_dir: str, raw_likelihood_map_A: Dict, 
                            raw_likelihood_map_B: Dict, world: World, sigma_value: float):
    """
    Create side-by-side comparison plots of old vs new smoothing methods for debugging.
    This function can be called during simulation to visualize smoothing differences.
    """
    logger = logging.getLogger(__name__)
    
    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    sigma_steps = max(1, int(sigma_value))
    
    old_smoothed_A = smooth_likelihoods_old(raw_likelihood_map_A, world, sigma_value)
    old_smoothed_B = smooth_likelihoods_old(raw_likelihood_map_B, world, sigma_value)
    
    precomputed_neighbors = compute_all_graph_neighbors(world, list(raw_likelihood_map_A.keys()))
    new_smoothed_A = smooth_likelihoods(raw_likelihood_map_A, sigma_steps, precomputed_neighbors)
    new_smoothed_B = smooth_likelihoods(raw_likelihood_map_B, sigma_steps, precomputed_neighbors)
    
    _create_smoothing_comparison_plot(
        trial_name, plots_dir, raw_likelihood_map_A, old_smoothed_A, new_smoothed_A, 
        world, "A", sigma_value, logger
    )
    
    _create_smoothing_comparison_plot(
        trial_name, plots_dir, raw_likelihood_map_B, old_smoothed_B, new_smoothed_B,
        world, "B", sigma_value, logger
    )


def _create_smoothing_comparison_plot(trial_name: str, plots_dir: str, raw_map: Dict, 
                                    old_smoothed: Dict, new_smoothed: Dict, world: World, 
                                    agent_id: str, sigma_value: float, logger):
    """Helper function to create a single smoothing comparison plot"""
    
    plot_width = world.width
    plot_height = world.height
    
    raw_grid = np.full((plot_height, plot_width), np.nan)
    old_grid = np.full((plot_height, plot_width), np.nan) 
    new_grid = np.full((plot_height, plot_width), np.nan)
    
    for coord, likelihood in raw_map.items():
        if 0 <= coord[0] < plot_width and 0 <= coord[1] < plot_height:
            raw_grid[coord[1], coord[0]] = likelihood
    
    for coord, likelihood in old_smoothed.items():
        if 0 <= coord[0] < plot_width and 0 <= coord[1] < plot_height:
            old_grid[coord[1], coord[0]] = likelihood
            
    for coord, likelihood in new_smoothed.items():
        if 0 <= coord[0] < plot_width and 0 <= coord[1] < plot_height:
            new_grid[coord[1], coord[0]] = likelihood
    
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(raw_grid, ax=axes[0], cmap='viridis', cbar=True, linewidths=0)
    axes[0].set_title(f"Raw Likelihoods - Agent {agent_id}")
    axes[0].set_xlabel("World X")
    axes[0].set_ylabel("World Y")
    
    sns.heatmap(old_grid, ax=axes[1], cmap='viridis', cbar=True, linewidths=0)
    axes[1].set_title(f"Grid-Based Smoothing - Agent {agent_id}\n(σ={sigma_value})")
    axes[1].set_xlabel("World X") 
    axes[1].set_ylabel("World Y")
    
    sns.heatmap(new_grid, ax=axes[2], cmap='viridis', cbar=True, linewidths=0)
    axes[2].set_title(f"Connectivity-Aware Smoothing - Agent {agent_id}\n(σ_steps={max(1, int(sigma_value))})")
    axes[2].set_xlabel("World X")
    axes[2].set_ylabel("World Y")
    
    plt.suptitle(f"Smoothing Comparison - Trial: {trial_name}, Agent {agent_id}", fontsize=16)
    plt.tight_layout()
    
    comparison_filename = os.path.join(plots_dir, f"smoothing_comparison_agent_{agent_id}_{trial_name}.png")
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved smoothing comparison plot for Agent {agent_id} to {comparison_filename}")
    plt.close()


def plot_suspect_paths_heatmap(trial_name: str, param_log_dir: str, agent_type_to_plot: str, 
                             evidence_type: str = 'visual', path_segment_type: str = 'return_from_fridge'):
    """
    Generates and saves heatmaps of sampled path frequencies for suspects A and B.
    """
    logger = logging.getLogger(__name__)

    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    
    plot_width = world.width
    plot_height = world.height
    
    csv_file_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_{agent_type_to_plot}.csv")

    paths_df = pd.read_csv(csv_file_path)

    # Determine which path segment to plot based on user request
    if path_segment_type == 'to_fridge':
        path_column_name = 'to_fridge_sequence'
        plot_title_segment = "Path to Fridge Tile Counts"
    elif path_segment_type == 'return_from_fridge':
        path_column_name = 'return_sequence'
        plot_title_segment = "Return Path Tile Counts"
    else:
        path_column_name = 'full_sequence'
        plot_title_segment = "Full Path Tile Counts"

    for agent_id in ['A', 'B']:
        agent_df = paths_df[paths_df['agent_id'] == agent_id]
        if agent_df.empty:
            continue

        agent_paths_str_list = agent_df[path_column_name].dropna().tolist()
        heatmap_grid = np.zeros((plot_height, plot_width))

        for path_str in agent_paths_str_list:
            try:
                # The path is already the correct segment, no need for complex slicing logic
                path_coords_to_plot = ast.literal_eval(path_str)
                for coord in path_coords_to_plot:
                    if isinstance(coord, (list, tuple)) and len(coord) == 2:
                        plot_x, plot_y = int(coord[0]), int(coord[1])
                        if 0 <= plot_x < plot_width and 0 <= plot_y < plot_height:
                            heatmap_grid[plot_y, plot_x] += 1
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse path string: {path_str}")
                continue

        plt.figure(figsize=(12, 8)) 
        cmap = 'Blues' if agent_id == 'A' else 'Greens'
        
        sns.heatmap(heatmap_grid, cmap=cmap, cbar=True, annot=True, fmt=".0f", 
                   linewidths=0.5, linecolor='gray')
        
        plt.title(f"{plot_title_segment} - Agent {agent_id}\n"
                 f"Trial: {trial_name}, Type: {agent_type_to_plot}")
        plt.xlabel("World X")
        plt.ylabel("World Y")
        
        plot_filename = os.path.join(plots_dir, 
                                   f"heatmap_{path_segment_type}_agent_{agent_id}_{trial_name}_{agent_type_to_plot}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved heatmap for Agent {agent_id} to {plot_filename}")
        plt.close()


def plot_multimodal_visual_predictions_heatmap(trial_name: str, param_log_dir: str, detective_agent_type: str):
    """Plots the visual component of multimodal detective predictions by averaging over audio sequences."""
    logger = logging.getLogger(__name__)
    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    predictions_file = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_multimodal_predictions.json")
    if not os.path.exists(predictions_file):
        logger.warning(f"Multimodal predictions file not found: {predictions_file}")
        return

    with open(predictions_file, 'r') as f:
        data = json.load(f)

    predictions_data = data.get('predictions', [])
    if not predictions_data:
        logger.warning(f"No predictions found in {predictions_file}")
        return

    # Aggregate predictions by crumb coordinate
    visual_predictions = {}
    for pred_entry in predictions_data:
        coord = tuple(pred_entry['crumb_coord'])
        prediction = pred_entry['prediction']
        if coord not in visual_predictions:
            visual_predictions[coord] = []
        visual_predictions[coord].append(prediction)

    # Average the predictions for each coordinate
    averaged_visual_predictions = {coord: np.mean(preds) for coord, preds in visual_predictions.items()}

    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    plot_width = world.width
    plot_height = world.height
    heatmap_grid = np.full((plot_height, plot_width), np.nan)
    
    for coord, avg_pred in averaged_visual_predictions.items():
        world_x, world_y = coord
        if 0 <= world_x < plot_width and 0 <= world_y < plot_height:
            heatmap_grid[world_y, world_x] = avg_pred

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_grid, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True, 
                vmin=-50, vmax=50, linewidths=.5, linecolor='gray')
    plt.title(f"Detective Predictions (Visual Component of Multimodal) - {detective_agent_type.capitalize()}\n"
              f"Trial: {trial_name}, Slider: (-50: A, 50: B)")
    plt.xlabel("Apartment X Coordinate")
    plt.ylabel("Apartment Y Coordinate")
    
    plt.xticks(ticks=np.arange(0.5, plot_width, 1), labels=np.arange(0, plot_width))
    plt.yticks(ticks=np.arange(0.5, plot_height, 1), labels=np.arange(0, plot_height))
    
    heatmap_filename = os.path.join(plots_dir, f"detective_multimodal_visual_preds_heatmap_{detective_agent_type}_{trial_name}.png")
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved multimodal visual predictions heatmap to {heatmap_filename}")
    plt.close()


def plot_multimodal_audio_predictions_heatmap(trial_name: str, param_log_dir: str, detective_agent_type: str):
    """Plots the audio component of multimodal detective predictions by averaging over crumb locations."""
    logger = logging.getLogger(__name__)
    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    predictions_file = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_multimodal_predictions.json")

    with open(predictions_file, 'r') as f:
        data = json.load(f)

    predictions_data = data.get('predictions', [])

    # Aggregate predictions by audio sequence length
    audio_predictions = {}
    for pred_entry in predictions_data:
        gt_sequence = pred_entry.get('gt_sequence', [])
        prediction = pred_entry.get('prediction')
        if len(gt_sequence) >= 5 and prediction is not None:
            len_key = (gt_sequence[0], gt_sequence[-1])
            if len_key not in audio_predictions:
                audio_predictions[len_key] = []
            audio_predictions[len_key].append(prediction)

    # Average the predictions for each audio length pair
    averaged_audio_predictions = {key: np.mean(preds) for key, preds in audio_predictions.items()}

    to_fridge_lengths = [key[0] for key in averaged_audio_predictions.keys()]
    from_fridge_lengths = [key[1] for key in averaged_audio_predictions.keys()]
    unique_to_lengths = sorted(set(to_fridge_lengths))
    unique_from_lengths = sorted(set(from_fridge_lengths), reverse=True)  # Reverse sort for y-axis
    
    heatmap_grid = np.full((len(unique_from_lengths), len(unique_to_lengths)), np.nan)
    
    to_length_map = {length: i for i, length in enumerate(unique_to_lengths)}
    from_length_map = {length: i for i, length in enumerate(unique_from_lengths)}
    
    for (to_len, from_len), avg_pred in averaged_audio_predictions.items():
        if to_len in to_length_map and from_len in from_length_map:
            row = from_length_map[from_len]
            col = to_length_map[to_len]
            heatmap_grid[row, col] = avg_pred

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_grid, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True,
                vmin=-50, vmax=50, linewidths=0.5, linecolor='gray',
                xticklabels=unique_to_lengths, yticklabels=unique_from_lengths)
    
    plt.title(f"Detective Predictions (Audio Component of Multimodal) - {detective_agent_type.capitalize()}\n"
              f"Trial: {trial_name}, Slider: (-50: A, +50: B)")
    plt.xlabel("Sequence Length TO Fridge")
    plt.ylabel("Sequence Length FROM Fridge")
    
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Detective Prediction (-50: A, +50: B)', rotation=270, labelpad=20)
    
    heatmap_filename = os.path.join(plots_dir, f"detective_multimodal_audio_preds_heatmap_{detective_agent_type}_{trial_name}.png")
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved multimodal audio predictions heatmap to {heatmap_filename}")
    plt.close()


def create_summary_plots(param_log_dir: str, trial_name: str):
    """Create summary plots for a simulation run"""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating summary plots for trial {trial_name} in {param_log_dir}")
    
    plot_types = [
        ('visual', 'return_from_fridge'),
        ('visual', 'to_fridge'), 
        ('audio', 'return_from_fridge'),
        ('audio', 'to_fridge')
    ]
    
    agent_types = ['naive', 'sophisticated']
    
    for agent_type in agent_types:
        csv_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_{agent_type}.csv")
        if os.path.exists(csv_path):
            for evidence_type, path_segment_type in plot_types:
                try:
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, 
                                             evidence_type, path_segment_type)
                except Exception as e:
                    logger.warning(f"Could not create {evidence_type} {path_segment_type} plot for {agent_type}: {e}")
        else:
            logger.warning(f"CSV file not found: {csv_path}")
    
    logger.info(f"Completed summary plots for {trial_name}")


def plot_detective_predictions_heatmap(trial_name: str, param_log_dir: str, detective_agent_type: str, evidence_type: str = 'visual'):
    """Plot detective predictions as a heatmap showing slider values (-50 to +50)"""
    logger = logging.getLogger(__name__)

    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if evidence_type == 'audio':
        plot_detective_audio_predictions_heatmap(trial_name, param_log_dir, detective_agent_type)
        return
    
    predictions_file = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_{evidence_type}_predictions.json")
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions_data = data.get('predictions', [])

    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    
    plot_width = world.width
    plot_height = world.height
    heatmap_grid = np.full((plot_height, plot_width), np.nan)
    
    predictions_plotted = 0
    for pred_entry in predictions_data:
        world_coord_tuple = None
        slider_prediction = None
        
        if "crumb_coord" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_coord"])
            slider_prediction = pred_entry.get("prediction")
        elif "crumb_location" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_location"])
            slider_prediction = pred_entry.get("slider_prediction", pred_entry.get("slider"))
        
        if world_coord_tuple and len(world_coord_tuple) == 2 and slider_prediction is not None:
            world_x, world_y = world_coord_tuple
            if 0 <= world_x < plot_width and 0 <= world_y < plot_height:
                heatmap_grid[world_y, world_x] = slider_prediction
                predictions_plotted += 1
    
    logger.info(f"Plotted {predictions_plotted} predictions on full apartment grid")
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_grid, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True, 
                vmin=-50, vmax=50, linewidths=.5, linecolor='gray')
    plt.title(f"Detective Predictions - Full Apartment ({detective_agent_type.capitalize()}, {evidence_type.capitalize()})\n"
              f"Trial: {trial_name}, Slider: (-50: A, 50: B)")
    plt.xlabel("Apartment X Coordinate")
    plt.ylabel("Apartment Y Coordinate")
    
    plt.xticks(ticks=np.arange(0.5, plot_width, 1), labels=np.arange(0, plot_width))
    plt.yticks(ticks=np.arange(0.5, plot_height, 1), labels=np.arange(0, plot_height))
    
    heatmap_filename = os.path.join(plots_dir, f"detective_preds_heatmap_{detective_agent_type}_{evidence_type}_{trial_name}.png")
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved detective predictions heatmap to {heatmap_filename}")
    plt.close()


def plot_detective_audio_predictions_heatmap(trial_name: str, param_log_dir: str, detective_agent_type: str):
    """Plot audio detective predictions as a heatmap based on sequence lengths to/from fridge"""
    logger = logging.getLogger(__name__)
    
    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    predictions_file = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_audio_predictions.json")
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions_data = data.get('predictions', [])
    
    to_fridge_lengths = []
    from_fridge_lengths = []
    prediction_values = []
    
    for pred_entry in predictions_data:
        gt_sequence = pred_entry.get('gt_sequence', [])
        prediction = pred_entry.get('prediction')
        
        if len(gt_sequence) >= 5 and prediction is not None:
            to_fridge = gt_sequence[0]
            from_fridge = gt_sequence[-1]
            
            to_fridge_lengths.append(to_fridge)
            from_fridge_lengths.append(from_fridge)
            prediction_values.append(prediction)
    
    unique_to_lengths = sorted(set(to_fridge_lengths))
    unique_from_lengths = sorted(set(from_fridge_lengths), reverse=True)  # Reverse sort for y-axis
    
    heatmap_grid = np.full((len(unique_from_lengths), len(unique_to_lengths)), np.nan)
    
    to_length_map = {length: i for i, length in enumerate(unique_to_lengths)}
    from_length_map = {length: i for i, length in enumerate(unique_from_lengths)}
    
    predictions_plotted = 0
    for to_len, from_len, pred_val in zip(to_fridge_lengths, from_fridge_lengths, prediction_values):
        if to_len in to_length_map and from_len in from_length_map:
            row = from_length_map[from_len]
            col = to_length_map[to_len]
            heatmap_grid[row, col] = pred_val
            predictions_plotted += 1
    
    logger.info(f"Plotted {predictions_plotted} audio predictions on sequence length grid")
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(heatmap_grid, 
                annot=True, 
                fmt=".1f", 
                cmap="coolwarm", 
                center=0, 
                cbar=True,
                vmin=-50, 
                vmax=50, 
                linewidths=0.5, 
                linecolor='gray',
                xticklabels=unique_to_lengths,
                yticklabels=unique_from_lengths
                )
    
    plt.title(f"Audio Detective Predictions ({detective_agent_type.capitalize()})\n"
              f"Trial: {trial_name}, Slider: (-50: A, +50: B)")
    plt.xlabel("Sequence Length TO Fridge")
    plt.ylabel("Sequence Length FROM Fridge")
    
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Detective Prediction (-50: A, +50: B)', rotation=270, labelpad=20)
    
    heatmap_filename = os.path.join(plots_dir, f"detective_audio_preds_heatmap_{detective_agent_type}_{trial_name}.png")
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved audio detective predictions heatmap to {heatmap_filename}")
    plt.close()


def plot_suspect_crumb_planting_heatmap(trial_name: str, param_log_dir: str):
    """Plot sophisticated suspect crumb planting locations"""
    logger = logging.getLogger(__name__)
    
    plots_dir = os.path.join(param_log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    csv_file_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_sophisticated.csv")
    
    paths_df = pd.read_csv(csv_file_path)
    
    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    
    plot_width = world.width
    plot_height = world.height
    
    for agent_id in ['A', 'B']:
        agent_df = paths_df[paths_df['agent_id'] == agent_id]
        
        plant_spots_str_list = agent_df['chosen_plant_spot'].dropna().tolist()
        plant_spots_counts = {}
        
        for spot_str in plant_spots_str_list:
            if spot_str and spot_str != 'None':
                try:
                    spot_coord = ast.literal_eval(spot_str)
                    if isinstance(spot_coord, (list, tuple)) and len(spot_coord) == 2:
                        spot_tuple = tuple(map(int, spot_coord))
                        plant_spots_counts[spot_tuple] = plant_spots_counts.get(spot_tuple, 0) + 1
                except (ValueError, SyntaxError):
                    continue
        
        heatmap_grid = np.zeros((plot_height, plot_width))
        
        for world_coord_tuple, count in plant_spots_counts.items():
            world_x, world_y = world_coord_tuple
            if 0 <= world_x < plot_width and 0 <= world_y < plot_height:
                heatmap_grid[world_y, world_x] = count
        
        plt.figure(figsize=(12, 8))
        cmap = 'Blues' if agent_id == 'A' else 'Reds'
        sns.heatmap(heatmap_grid, annot=True, fmt=".0f", cmap=cmap, cbar=True, 
                   linewidths=0.5, linecolor='gray')
        plt.title(f"Agent {agent_id}: Sophisticated Crumb Planting Locations\n"
                 f"Trial: {trial_name} (Full Apartment View)")
        plt.xlabel("World X")
        plt.ylabel("World Y")
        
        plot_filename = os.path.join(plots_dir, f"crumb_planting_heatmap_agent_{agent_id}_{trial_name}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved crumb planting heatmap for Agent {agent_id} to {plot_filename}")
        plt.close()


def create_simulation_plots(param_log_dir: str, trial_name: str, evidence_type: str):
    """Create comprehensive plots for a simulation run based on evidence type"""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating {evidence_type} evidence plots for trial {trial_name} in {param_log_dir}")
    
    path_segments = ['return_from_fridge', 'to_fridge']
    agent_types = ['naive', 'sophisticated']
    
    for agent_type in agent_types:
        csv_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_{agent_type}.csv")
        for path_segment_type in path_segments:
            try:
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, 
                                             'visual', path_segment_type)
            except Exception as e:
                logger.warning(f"Could not create {path_segment_type} plot for {agent_type}: {e}")
    
    if evidence_type in ['visual', 'multimodal']:
        try:
            plot_suspect_crumb_planting_heatmap(trial_name, param_log_dir)
        except Exception as e:
            logger.warning(f"Could not create crumb planting heatmap: {e}")
    
    for agent_type in ['naive', 'sophisticated']:
        if evidence_type == 'multimodal':
            try:
                plot_multimodal_visual_predictions_heatmap(trial_name, param_log_dir, agent_type)
            except Exception as e:
                logger.warning(f"Could not create multimodal visual predictions plot for {agent_type}: {e}")
            try:
                plot_multimodal_audio_predictions_heatmap(trial_name, param_log_dir, agent_type)
            except Exception as e:
                logger.warning(f"Could not create multimodal audio predictions plot for {agent_type}: {e}")
        
        elif evidence_type == 'visual':
            try:
                plot_detective_predictions_heatmap(trial_name, param_log_dir, agent_type, evidence_type='visual')
            except Exception as e:
                logger.warning(f"Could not create visual detective predictions plot for {agent_type}: {e}")

        elif evidence_type == 'audio':
            try:
                plot_detective_audio_predictions_heatmap(trial_name, param_log_dir, agent_type)
            except Exception as e:
                logger.warning(f"Could not create audio detective predictions plot for {agent_type}: {e}")

    logger.info(f"Completed {evidence_type} evidence plots for {trial_name}")
