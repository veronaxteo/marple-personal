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
from typing import Dict, Tuple, List
from ..core.world import World
from ..utils.math_utils import smooth_likelihood_grid, smooth_likelihood_grid_connectivity_aware


def plot_smoothing_comparison(trial_name: str, param_log_dir: str, raw_likelihood_map_A: Dict, 
                            raw_likelihood_map_B: Dict, world: World, sigma_value: float):
    """
    Create side-by-side comparison plots of old vs new smoothing methods for debugging.
    This function can be called during simulation to visualize smoothing differences.
    """
    logger = logging.getLogger(__name__)
    
    # Apply both smoothing methods
    sigma_steps = max(1, int(sigma_value))
    
    # 2d grid-based smoothing
    old_smoothed_A = smooth_likelihood_grid(raw_likelihood_map_A, world, sigma_value)
    old_smoothed_B = smooth_likelihood_grid(raw_likelihood_map_B, world, sigma_value)
    
    # Connectivity-aware smoothing
    new_smoothed_A = smooth_likelihood_grid_connectivity_aware(raw_likelihood_map_A, world, sigma_steps)
    new_smoothed_B = smooth_likelihood_grid_connectivity_aware(raw_likelihood_map_B, world, sigma_steps)
    
    # Create comparison plots for Agent A
    _create_smoothing_comparison_plot(
        trial_name, param_log_dir, raw_likelihood_map_A, old_smoothed_A, new_smoothed_A, 
        world, "A", sigma_value, logger
    )
    
    # Create comparison plots for Agent B  
    _create_smoothing_comparison_plot(
        trial_name, param_log_dir, raw_likelihood_map_B, old_smoothed_B, new_smoothed_B,
        world, "B", sigma_value, logger
    )


def _create_smoothing_comparison_plot(trial_name: str, param_log_dir: str, raw_map: Dict, 
                                    old_smoothed: Dict, new_smoothed: Dict, world: World, 
                                    agent_id: str, sigma_value: float, logger):
    """Helper function to create a single smoothing comparison plot"""
    
    plot_width = world.width
    plot_height = world.height
    
    # Create grids for each smoothing method
    raw_grid = np.full((plot_height, plot_width), np.nan)
    old_grid = np.full((plot_height, plot_width), np.nan) 
    new_grid = np.full((plot_height, plot_width), np.nan)
    
    # Fill raw likelihood grid
    for coord, likelihood in raw_map.items():
        if 0 <= coord[0] < plot_width and 0 <= coord[1] < plot_height:
            raw_grid[coord[1], coord[0]] = likelihood
    
    # Fill old smoothed grid
    for coord, likelihood in old_smoothed.items():
        if 0 <= coord[0] < plot_width and 0 <= coord[1] < plot_height:
            old_grid[coord[1], coord[0]] = likelihood
            
    # Fill new smoothed grid
    for coord, likelihood in new_smoothed.items():
        if 0 <= coord[0] < plot_width and 0 <= coord[1] < plot_height:
            new_grid[coord[1], coord[0]] = likelihood
    
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Raw likelihoods
    sns.heatmap(raw_grid, ax=axes[0], cmap='viridis', cbar=True, linewidths=0)
    axes[0].set_title(f"Raw Likelihoods - Agent {agent_id}")
    axes[0].set_xlabel("World X")
    axes[0].set_ylabel("World Y")
    
    # Old grid-based smoothing
    sns.heatmap(old_grid, ax=axes[1], cmap='viridis', cbar=True, linewidths=0)
    axes[1].set_title(f"Grid-Based Smoothing - Agent {agent_id}\n(σ={sigma_value})")
    axes[1].set_xlabel("World X") 
    axes[1].set_ylabel("World Y")
    
    # New connectivity-aware smoothing
    sns.heatmap(new_grid, ax=axes[2], cmap='viridis', cbar=True, linewidths=0)
    axes[2].set_title(f"Connectivity-Aware Smoothing - Agent {agent_id}\n(σ_steps={max(1, int(sigma_value))})")
    axes[2].set_xlabel("World X")
    axes[2].set_ylabel("World Y")
    
    plt.suptitle(f"Smoothing Comparison - Trial: {trial_name}, Agent {agent_id}", fontsize=16)
    plt.tight_layout()
    
    # Save plot
    comparison_filename = os.path.join(param_log_dir, f"smoothing_comparison_agent_{agent_id}_{trial_name}.png")
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved smoothing comparison plot for Agent {agent_id} to {comparison_filename}")
    plt.close()


def plot_suspect_paths_heatmap(trial_name: str, param_log_dir: str, agent_type_to_plot: str, 
                             evidence_type: str = 'visual', path_segment_type: str = 'return_from_fridge'):
    """
    Generates and saves heatmaps of sampled path frequencies for suspects A and B.
    
    Args:
        trial_name: Name of the trial
        param_log_dir: Directory containing the CSV files
        agent_type_to_plot: Agent type ('naive' or 'sophisticated')
        evidence_type: Evidence type ('visual' or 'audio')
        path_segment_type: Path segment ('to_fridge', 'return_from_fridge')
    """
    logger = logging.getLogger(__name__)

    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    
    # Use full apartment dimensions for both audio and visual evidence
    plot_width = world.width
    plot_height = world.height
    coord_offset_x = 0
    coord_offset_y = 0

    csv_file_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_{agent_type_to_plot}.csv")
    paths_df = pd.read_csv(csv_file_path)

    # Determine path column and title based on evidence type and segment type
    if path_segment_type == 'to_fridge':
        if evidence_type == 'audio':
            path_column_name = 'to_fridge_sequence_world_coords'
        elif evidence_type == 'visual':
            path_column_name = 'full_sequence_world_coords'
        plot_title_segment = "Path to Fridge Tile Counts"
    elif path_segment_type == 'return_from_fridge':
        if evidence_type == 'audio':
            path_column_name = 'middle_sequence_world_coords'  # From fridge to door
            plot_title_segment = "Path from Fridge to Door Tile Counts"
        elif evidence_type == 'visual':
            path_column_name = 'full_sequence_world_coords'  # Will slice from fridge
            plot_title_segment = "Return Path Tile Counts (from Fridge)"
    else:
        path_column_name = 'full_sequence_world_coords'
        plot_title_segment = "Full Path Tile Counts"

    fridge_access_point = world.get_fridge_access_point()

    for agent_id in ['A', 'B']:
        agent_df = paths_df[paths_df['agent'] == agent_id]

        agent_paths_str_list = agent_df[path_column_name].tolist()
        heatmap_grid = np.zeros((plot_height, plot_width))
        paths_processed_count = 0

        for path_str in agent_paths_str_list:
            path_coords_world_full = ast.literal_eval(path_str)
            
            path_coords_to_plot = []
            if evidence_type == 'visual' and fridge_access_point and path_coords_world_full:
                # Visual: handle path slicing based on segment type
                fridge_ap_tuple = tuple(fridge_access_point)
                path_coords_world_tuples = [tuple(map(int, coord)) for coord in path_coords_world_full if isinstance(coord, (list, tuple)) and len(coord) == 2]
                fridge_idx = -1
                for i, coord_tuple in enumerate(path_coords_world_tuples):
                    if coord_tuple == fridge_ap_tuple:
                        fridge_idx = i
                        break
                
                if fridge_idx != -1:
                    if path_segment_type == 'to_fridge':
                        # Plot from start to fridge (inclusive)
                        path_coords_to_plot = path_coords_world_tuples[:fridge_idx + 1]
                    elif path_segment_type == 'return_from_fridge':
                        # Plot from fridge to end
                        path_coords_to_plot = path_coords_world_tuples[fridge_idx:]
                    else:
                        path_coords_to_plot = path_coords_world_tuples
                else:
                    path_coords_to_plot = path_coords_world_tuples
            elif evidence_type == 'audio':
                # Audio: plot the selected segment directly using full apartment
                path_coords_to_plot = [tuple(map(int, coord)) for coord in path_coords_world_full if isinstance(coord, (list, tuple)) and len(coord) == 2]
            else:
                path_coords_to_plot = [tuple(map(int, coord)) for coord in path_coords_world_full if isinstance(coord, (list, tuple)) and len(coord) == 2]

            paths_processed_count += 1
            for world_coord_tuple_anytype in path_coords_to_plot: 
                if isinstance(world_coord_tuple_anytype, (list, tuple)) and len(world_coord_tuple_anytype) == 2 and \
                   all(isinstance(c, (int, float)) for c in world_coord_tuple_anytype):
                    world_coord_tuple = tuple(map(int, world_coord_tuple_anytype))
                    
                    # For both audio and visual, plot all coordinates within apartment bounds
                    plot_x = world_coord_tuple[0] - coord_offset_x
                    plot_y = world_coord_tuple[1] - coord_offset_y
                    if 0 <= plot_x < plot_width and 0 <= plot_y < plot_height:
                        heatmap_grid[plot_y, plot_x] += 1

        plt.figure(figsize=(12, 8)) 
        
        # Set color based on agent_id
        cmap = 'Reds' if agent_id == 'A' else 'Blues'
        
        # Create heatmap
        sns.heatmap(heatmap_grid, cmap=cmap, cbar=True, linewidths=0)
        
        # Add labels and title
        plt.title(f"{plot_title_segment} - Agent {agent_id}\n"
                 f"Trial: {trial_name}, Type: {agent_type_to_plot}, Evidence: {evidence_type}")
        plt.xlabel("World X")
        plt.ylabel("World Y")
        
        # Save plot
        plot_filename = os.path.join(param_log_dir, 
                                   f"heatmap_{evidence_type}_{path_segment_type}_agent_{agent_id}_{trial_name}_{agent_type_to_plot}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved heatmap for Agent {agent_id} to {plot_filename}")
        plt.close()


def create_summary_plots(param_log_dir: str, trial_name: str):
    """Create summary plots for a simulation run"""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating summary plots for trial {trial_name} in {param_log_dir}")
    
    # List of potential plot types to generate
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
    
    # Load prediction data
    predictions_file = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_{evidence_type}_predictions.json")
    if not os.path.exists(predictions_file):
        logger.warning(f"Predictions file not found: {predictions_file}")
        return
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions_data = data.get('predictions', [])
    if not predictions_data:
        logger.warning(f"No predictions found in {predictions_file}")
        return
    
    # Load world for dimensions
    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    
    # Create full apartment view
    plot_width = world.width
    plot_height = world.height
    heatmap_grid = np.full((plot_height, plot_width), np.nan)
    
    predictions_plotted = 0
    for pred_entry in predictions_data:
        # Handle both data structures
        world_coord_tuple = None
        slider_prediction = None
        
        if "crumb_coord" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_coord"])
            slider_prediction = pred_entry.get("prediction")
        elif "crumb_location_world_coords" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_location_world_coords"])
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
    
    # Set ticks for world coordinates
    plt.xticks(ticks=np.arange(0.5, plot_width, 1), labels=np.arange(1, plot_width + 1))
    plt.yticks(ticks=np.arange(0.5, plot_height, 1), labels=np.arange(plot_height, 0, -1))
    
    heatmap_filename = os.path.join(param_log_dir, f"detective_preds_heatmap_{detective_agent_type}_{evidence_type}_{trial_name}.png")
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved detective predictions heatmap to {heatmap_filename}")
    plt.close()


def plot_suspect_crumb_planting_heatmap(trial_name: str, param_log_dir: str):
    """Plot sophisticated suspect crumb planting locations"""
    logger = logging.getLogger(__name__)
    
    # Load sophisticated paths CSV
    csv_file_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_sophisticated.csv")
    if not os.path.exists(csv_file_path):
        logger.warning(f"Sophisticated paths CSV not found: {csv_file_path}")
        return
    
    paths_df = pd.read_csv(csv_file_path)
    
    # Load world for coordinates
    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    
    kitchen_width = world.coordinate_mapper.kitchen_width
    kitchen_height = world.coordinate_mapper.kitchen_height
    
    for agent_id in ['A', 'B']:
        agent_df = paths_df[paths_df['agent'] == agent_id]
        
        # Extract chosen plant spots
        plant_spots_str_list = agent_df['chosen_plant_spot'].tolist()
        plant_spots_counts = {}
        
        for spot_str in plant_spots_str_list:
            if spot_str and spot_str != 'None' and spot_str != '[]':
                try:
                    spot_coord = ast.literal_eval(spot_str)
                    if isinstance(spot_coord, (list, tuple)) and len(spot_coord) == 2:
                        spot_tuple = tuple(spot_coord)
                        plant_spots_counts[spot_tuple] = plant_spots_counts.get(spot_tuple, 0) + 1
                except (ValueError, SyntaxError):
                    continue
        
        if not plant_spots_counts:
            logger.warning(f"No valid plant spots found for Agent {agent_id}")
            continue
        
        # Create heatmap grid
        heatmap_grid = np.zeros((kitchen_height, kitchen_width))
        
        for world_coord_tuple, count in plant_spots_counts.items():
            kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
            if kitchen_coord:
                kx, ky = kitchen_coord
                if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                    heatmap_grid[ky, kx] = count
        
        plt.figure(figsize=(10, 4))
        cmap = 'Reds' if agent_id == 'A' else 'Blues'
        sns.heatmap(heatmap_grid, annot=True, fmt=".0f", cmap=cmap, cbar=True, linewidths=.5, linecolor='gray')
        plt.title(f"Agent {agent_id}: Sophisticated Crumb Planting Locations\nTrial: {trial_name}")
        plt.xlabel("Kitchen X")
        plt.ylabel("Kitchen Y")
        
        plt.xticks(ticks=np.arange(0.5, kitchen_width, 1), labels=np.arange(1, kitchen_width + 1))
        plt.yticks(ticks=np.arange(0.5, kitchen_height, 1), labels=np.arange(kitchen_height, 0, -1))
        
        plot_filename = os.path.join(param_log_dir, f"crumb_planting_heatmap_agent_{agent_id}_{trial_name}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved crumb planting heatmap for Agent {agent_id} to {plot_filename}")
        plt.close()


def create_simulation_plots(param_log_dir: str, trial_name: str, evidence_type: str):
    """Create comprehensive plots for a simulation run based on evidence type"""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating {evidence_type} evidence plots for trial {trial_name} in {param_log_dir}")
    
    # Path segment types to plot
    path_segments = ['return_from_fridge', 'to_fridge']
    agent_types = ['naive', 'sophisticated']
    
    # 1. Plot suspect path heatmaps (only for the evidence type used)
    for agent_type in agent_types:
        csv_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_{agent_type}.csv")
        if os.path.exists(csv_path):
            for path_segment_type in path_segments:
                try:
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, 
                                             evidence_type, path_segment_type)
                except Exception as e:
                    logger.warning(f"Could not create {evidence_type} {path_segment_type} plot for {agent_type}: {e}")
        else:
            logger.warning(f"CSV file not found: {csv_path}")
    
    # 2. Plot crumb planting heatmaps (for sophisticated agents in visual evidence)
    if evidence_type == 'visual':
        try:
            plot_suspect_crumb_planting_heatmap(trial_name, param_log_dir)
        except Exception as e:
            logger.warning(f"Could not create crumb planting heatmap: {e}")
    
    # 3. Plot detective predictions for both naive and sophisticated
    for agent_type in ['naive', 'sophisticated']:
        try:
            plot_detective_predictions_heatmap(trial_name, param_log_dir, agent_type, evidence_type)
        except Exception as e:
            logger.warning(f"Could not create detective predictions plot for {agent_type}: {e}")
    
    logger.info(f"Completed {evidence_type} evidence plots for {trial_name}") 
