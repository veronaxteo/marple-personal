import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import logging
import ast
import argparse
import sys
import glob
import json
from collections import deque
from src.world import World
from src.utils.math_utils import _get_graph_neighbors, smooth_likelihood_grid, smooth_likelihood_grid_connectivity_aware


def plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type_to_plot, evidence_type='visual', path_segment_type='return_from_fridge'):
    """
    Generates and saves heatmaps of sampled path frequencies for suspects A and B.
    evidence_type: 'visual' or 'audio'
    path_segment_type: 'to_fridge', 'return_from_fridge' (for visual and audio both support both types)
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
            path_column_name = 'full_sequence_world_coords'  # Will slice to fridge
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
                    # Fallback if fridge not found
                    path_coords_to_plot = path_coords_world_tuples
            elif evidence_type == 'audio':
                # Audio: plot the selected segment directly using full apartment
                path_coords_to_plot = [tuple(map(int, coord)) for coord in path_coords_world_full if isinstance(coord, (list, tuple)) and len(coord) == 2]
            else:
                # Fallback
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

        plt.figure(figsize=(12, 8))  # Use same size for both evidence types
        
        # Set color based on agent_id
        plot_cmap = "Blues" if agent_id == 'A' else "Greens"
        
        sns.heatmap(heatmap_grid, annot=True, fmt=".0f", cmap=plot_cmap, cbar=True, linewidths=.5, linecolor='gray')
        
        plt.title(f"Agent {agent_id} ({agent_type_to_plot.capitalize()}): {plot_title_segment} \n Trial: {trial_name} (Apartment Coordinates)")
        plt.xlabel("Apartment X")
        plt.ylabel("Apartment Y")
        
        plt.xticks(ticks=np.arange(0.5, plot_width, 1), labels=np.arange(1, plot_width + 1))
        plt.yticks(ticks=np.arange(0.5, plot_height, 1), labels=np.arange(plot_height, 0, -1))

        heatmap_filename = os.path.join(param_log_dir, f"sampled_paths_heatmap_agent_{agent_id}_{agent_type_to_plot}_{path_segment_type}_{trial_name}.png")
        try:
            plt.savefig(heatmap_filename)
            logger.info(f"Saved path heatmap for Agent {agent_id} ({agent_type_to_plot}) to {heatmap_filename}")
        except Exception as e:
            logger.error(f"Error saving path heatmap for Agent {agent_id} ({agent_type_to_plot}): {e}")
        plt.close()


def plot_suspect_crumb_planting_heatmap(trial_name, param_log_dir):
    """
    Generates and saves heatmaps of chosen crumb planting locations for sophisticated suspects A and B.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO) 

    # Load world object to get kitchen dimensions
    trial_file_name = f"{trial_name}_A1.json" 
    world = World.initialize_world_start(trial_file_name)
    logger.info(f"Successfully loaded world for trial: {trial_name}")

    kitchen_width = world.coordinate_mapper.kitchen_width
    kitchen_height = world.coordinate_mapper.kitchen_height

    # Load chosen plant spots data from CSV
    csv_file_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_sophisticated.csv")
    if not os.path.exists(csv_file_path):
        logger.error(f"Sophisticated agent CSV file not found: {csv_file_path}. Cannot generate heatmaps.")
        return

    soph_paths_df = pd.read_csv(csv_file_path)
    logger.info(f"Loaded sophisticated paths from {csv_file_path}")

    agent_A_spots_str = soph_paths_df[soph_paths_df['agent'] == 'A']['chosen_plant_spot_world_coords'].tolist()
    agent_B_spots_str = soph_paths_df[soph_paths_df['agent'] == 'B']['chosen_plant_spot_world_coords'].tolist()

    agent_A_spots = [ast.literal_eval(s) for s in agent_A_spots_str]
    agent_B_spots = [ast.literal_eval(s) for s in agent_B_spots_str]

    for agent_id, agent_spots in [('A', agent_A_spots), ('B', agent_B_spots)]:
        if not agent_spots:
            logger.info(f"No plant spots found for Agent {agent_id}. Skipping heatmap.")
            continue

        valid_spots = [tuple(spot) for spot in agent_spots if spot is not None]
        spot_counts = Counter(valid_spots)

        # Create heatmap grid (kitchen coordinates)
        heatmap_grid = np.zeros((kitchen_height, kitchen_width))

        for world_coord_tuple, count in spot_counts.items():
            kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
            kx, ky = kitchen_coord # kx = col, ky = row
            if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                heatmap_grid[ky, kx] = count

        plt.figure(figsize=(10, 4))
        sns.heatmap(heatmap_grid, annot=True, fmt=".0f", cmap="Reds", cbar=True, linewidths=.5, linecolor='gray')
        plt.title(f"Agent {agent_id} (Sophisticated): Chosen Crumb Planting Locations\nTrial: {trial_name}")
        plt.xlabel("Kitchen X")
        plt.ylabel("Kitchen Y")
        
        # Set x and y ticks to start at 1
        plt.xticks(ticks=np.arange(0.5, kitchen_width, 1), labels=np.arange(1, kitchen_width + 1))
        plt.yticks(ticks=np.arange(0.5, kitchen_height, 1), labels=np.arange(kitchen_height, 0, -1))

        heatmap_filename = os.path.join(param_log_dir, f"soph_planting_heatmap_agent_{agent_id}_{trial_name}.png")
        try:
            plt.savefig(heatmap_filename)
            logger.info(f"Saved heatmap for Agent {agent_id} to {heatmap_filename}")
        except Exception as e:
            logger.error(f"Error saving heatmap for Agent {agent_id}: {e}")
        plt.close()


def plot_detective_predictions_heatmap(trial_name, param_log_dir, detective_agent_type, evidence_type='visual'):
    """
    Generates and saves a heatmap of detective slider predictions for each crumb location.
    For visual evidence, uses world coordinates to show full apartment context for debugging smoothing.
    For audio evidence, this function may not be applicable as audio evidence doesn't have spatial crumb locations.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    trial_file_name = f"{trial_name}_A1.json" 
    world = World.initialize_world_start(trial_file_name)
    logger.info(f"Successfully loaded world for trial: {trial_name} to plot detective predictions.")
    
    # Detective predictions are typically for visual evidence with crumb locations
    # For audio evidence, this heatmap doesn't make sense as predictions are about audio sequences, not spatial locations
    if evidence_type == 'audio':
        logger.info(f"Skipping detective predictions heatmap for audio evidence (no spatial crumb locations)")
        return
    
    # Use full apartment dimensions to debug smoothing effects
    plot_width = world.width
    plot_height = world.height
    logger.info(f"Plotting detective predictions on full apartment grid: {plot_width}x{plot_height}")

    # Updated to new naming convention: {trial_name}_{agent_type}_{evidence}_predictions.json
    json_file_path = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_{evidence_type}_predictions.json")
    
    if not os.path.exists(json_file_path):
        logger.warning(f"Prediction file not found: {json_file_path}")
        return

    with open(json_file_path, 'r') as f:
        predictions_data = json.load(f)
    logger.info(f"Loaded detective predictions from {json_file_path}")

    # Initialize with NaN for missing spots - use full apartment dimensions
    heatmap_grid = np.full((plot_height, plot_width), np.nan) 

    predictions_plotted = 0
    for pred_entry in predictions_data:
        # Handle both old and new data structures
        world_coord_tuple = None
        slider_prediction = None
        
        # Try new data structure first (from VisualEvidenceProcessor)
        if "crumb_coord" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_coord"])
            slider_prediction = pred_entry.get("prediction")
        # Fallback to old data structure
        elif "crumb_location_world_coords" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_location_world_coords"])
            slider_prediction = pred_entry.get("slider_prediction", pred_entry.get("slider"))
        
        if world_coord_tuple and len(world_coord_tuple) == 2 and slider_prediction is not None:
            # Use world coordinates directly instead of converting to kitchen coordinates
            world_x, world_y = world_coord_tuple
            
            # Check bounds for full apartment
            if 0 <= world_x < plot_width and 0 <= world_y < plot_height:
                heatmap_grid[world_y, world_x] = slider_prediction
                predictions_plotted += 1
    
    logger.info(f"Plotted {predictions_plotted} predictions on full apartment grid")

    plt.figure(figsize=(14, 10))  # Larger figure for full apartment view
    sns.heatmap(heatmap_grid, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True, vmin=-50, vmax=50, linewidths=.5, linecolor='gray')
    plt.title(f"Detective Predictions - Full Apartment View ({detective_agent_type.capitalize()}, {evidence_type.capitalize()})\nTrial: {trial_name}, Slider: (-50: A, 50: B)")
    plt.xlabel("Apartment X Coordinate")
    plt.ylabel("Apartment Y Coordinate") 
    
    # Set ticks for world coordinates - match the suspect path heatmap formatting exactly
    plt.xticks(ticks=np.arange(0.5, plot_width, 1), labels=np.arange(1, plot_width + 1))
    plt.yticks(ticks=np.arange(0.5, plot_height, 1), labels=np.arange(plot_height, 0, -1))

    heatmap_filename = os.path.join(param_log_dir, f"detective_preds_heatmap_fullworld_{detective_agent_type}_{evidence_type}_{trial_name}.png")
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')  # Higher DPI for better resolution
    logger.info(f"Saved detective predictions heatmap (full world view) to {heatmap_filename}")
    plt.close()

    # Also create the original kitchen-only view for comparison
    _plot_detective_predictions_kitchen_only(trial_name, param_log_dir, detective_agent_type, evidence_type, world, predictions_data, logger)


def _plot_detective_predictions_kitchen_only(trial_name, param_log_dir, detective_agent_type, evidence_type, world, predictions_data, logger):
    """Helper function to create the original kitchen-only detective predictions plot for comparison"""
    
    kitchen_width = world.coordinate_mapper.kitchen_width
    kitchen_height = world.coordinate_mapper.kitchen_height
    
    heatmap_grid = np.full((kitchen_height, kitchen_width), np.nan)

    for pred_entry in predictions_data:
        world_coord_tuple = None
        slider_prediction = None
        
        if "crumb_coord" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_coord"])
            slider_prediction = pred_entry.get("prediction")
        elif "crumb_location_world_coords" in pred_entry:
            world_coord_tuple = tuple(pred_entry["crumb_location_world_coords"])
            slider_prediction = pred_entry.get("slider_prediction", pred_entry.get("slider"))
        
        if world_coord_tuple and len(world_coord_tuple) == 2 and slider_prediction is not None:
            kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
            if kitchen_coord:
                kx, ky = kitchen_coord
                if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                    heatmap_grid[ky, kx] = slider_prediction

    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_grid, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True, vmin=-50, vmax=50, linewidths=.5, linecolor='gray')
    plt.title(f"Detective Predictions - Kitchen Only ({detective_agent_type.capitalize()}, {evidence_type.capitalize()})\nTrial: {trial_name}, Slider: (-50: A, 50: B)")
    plt.xlabel("Kitchen X")
    plt.ylabel("Kitchen Y")
    
    plt.xticks(ticks=np.arange(0.5, kitchen_width, 1), labels=np.arange(1, kitchen_width + 1))
    plt.yticks(ticks=np.arange(0.5, kitchen_height, 1), labels=np.arange(kitchen_height, 0, -1))

    heatmap_filename = os.path.join(param_log_dir, f"detective_preds_heatmap_kitchen_{detective_agent_type}_{evidence_type}_{trial_name}.png")
    plt.savefig(heatmap_filename)
    logger.info(f"Saved detective predictions heatmap (kitchen only) to {heatmap_filename}")
    plt.close()


def plot_smoothing_comparison(trial_name, param_log_dir, raw_likelihood_map_A, raw_likelihood_map_B, world, sigma_value):
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


def _create_smoothing_comparison_plot(trial_name, param_log_dir, raw_map, old_smoothed, new_smoothed, world, agent_id, sigma_value, logger):
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
    
    # Create side-by-side comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
    
    # Save comparison plot
    comparison_filename = os.path.join(param_log_dir, f"smoothing_comparison_agent_{agent_id}_{trial_name}.png")
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved smoothing comparison plot for Agent {agent_id} to {comparison_filename}")
    plt.close()


def plot_retrospective_smoothing_comparison(trial_name, param_log_dir):
    """
    Generate smoothing comparison plots by reconstructing raw likelihoods from prediction data.
    This allows debugging smoothing effects after simulation has completed.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load world and prediction data
        trial_file_name = f"{trial_name}_A1.json"
        world = World.initialize_world_start(trial_file_name)
        
        # Load sophisticated predictions (these use smoothed likelihoods)
        soph_json_path = os.path.join(param_log_dir, f"{trial_name}_sophisticated_visual_predictions.json")
        if not os.path.exists(soph_json_path):
            logger.warning(f"Sophisticated predictions file not found: {soph_json_path}")
            return
            
        with open(soph_json_path, 'r') as f:
            soph_predictions = json.load(f)
        
        # Load naive predictions (these should be raw, unsmoothed likelihoods)
        naive_json_path = os.path.join(param_log_dir, f"{trial_name}_naive_visual_predictions.json")
        if not os.path.exists(naive_json_path):
            logger.warning(f"Naive predictions file not found: {naive_json_path}")
            return
            
        with open(naive_json_path, 'r') as f:
            naive_predictions = json.load(f)
        
        # Reconstruct raw likelihood maps from naive predictions
        raw_likelihood_map_A = {}
        raw_likelihood_map_B = {}
        
        for pred_entry in naive_predictions:
            world_coord_tuple = None
            likelihood_A = None
            likelihood_B = None
            
            # Handle different data structures
            if "crumb_coord" in pred_entry:
                world_coord_tuple = tuple(pred_entry["crumb_coord"])
                likelihood_A = pred_entry.get("likelihood_A")
                likelihood_B = pred_entry.get("likelihood_B")
            elif "crumb_location_world_coords" in pred_entry:
                world_coord_tuple = tuple(pred_entry["crumb_location_world_coords"])
                likelihood_A = pred_entry.get("evidence_likelihood_A")
                likelihood_B = pred_entry.get("evidence_likelihood_B")
            
            if world_coord_tuple and likelihood_A is not None and likelihood_B is not None:
                raw_likelihood_map_A[world_coord_tuple] = likelihood_A
                raw_likelihood_map_B[world_coord_tuple] = likelihood_B
        
        if not raw_likelihood_map_A or not raw_likelihood_map_B:
            logger.warning("Could not reconstruct raw likelihood maps from naive predictions")
            return
        
        # Read metadata to get sigma value
        metadata_path = os.path.join(param_log_dir, 'metadata.json')
        sigma_value = 1.0  # Default fallback
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Try both possible locations for sigma value
            sigma_value = metadata.get('soph_detective_sigma', 
                                     metadata.get('parameters', {}).get('soph_detective_sigma', 1.0))
                
        # Generate comparison plots
        plot_smoothing_comparison(trial_name, param_log_dir, raw_likelihood_map_A, raw_likelihood_map_B, world, sigma_value)
        
    except Exception as e:
        logger.error(f"Error generating retrospective smoothing comparison: {e}")
        import traceback
        traceback.print_exc()


def plot_crumb_planting_comparison(trial_name, param_log_dir, detective_agent_type, evidence_type='visual'):
    """
    Creates side-by-side comparison of actual sophisticated crumb planting vs detective simulated counts.
    Only works for visual evidence with spatial crumb locations.
    """
    logger = logging.getLogger(__name__)
    
    if evidence_type != 'visual':
        logger.info(f"Skipping crumb planting comparison for non-visual evidence type: {evidence_type}")
        return
    
    # Check if we have both sophisticated suspect data and detective predictions
    soph_csv_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_sophisticated.csv")
    detective_json_path = os.path.join(param_log_dir, f"{trial_name}_{detective_agent_type}_{evidence_type}_predictions.json")
    
    if not os.path.exists(soph_csv_path):
        logger.warning(f"Sophisticated suspect CSV not found: {soph_csv_path}")
        return
        
    if not os.path.exists(detective_json_path):
        logger.warning(f"Detective predictions JSON not found: {detective_json_path}")
        return
    
    # Load world
    trial_file_name = f"{trial_name}_A1.json" 
    world = World.initialize_world_start(trial_file_name)
    kitchen_width = world.coordinate_mapper.kitchen_width
    kitchen_height = world.coordinate_mapper.kitchen_height
    
    # Load sophisticated agent actual planting data
    soph_paths_df = pd.read_csv(soph_csv_path)
    
    # Load detective predictions
    with open(detective_json_path, 'r') as f:
        predictions_data = json.load(f)
    
    # Process data for each agent
    for agent_id in ['A', 'B']:
        # Get actual sophisticated planting data
        agent_spots_str = soph_paths_df[soph_paths_df['agent'] == agent_id]['chosen_plant_spot_world_coords'].tolist()
        agent_spots = [ast.literal_eval(s) for s in agent_spots_str if s != 'None']
        actual_spots_counts = Counter([tuple(spot) for spot in agent_spots if spot is not None])
        
        # Get detective simulated counts
        detective_likelihoods = {}
        for pred_entry in predictions_data:
            world_coord_tuple = None
            likelihood = None
            
            if "crumb_coord" in pred_entry:
                world_coord_tuple = tuple(pred_entry["crumb_coord"])
                likelihood = pred_entry.get(f"likelihood_{agent_id}")
            elif "crumb_location_world_coords" in pred_entry:
                world_coord_tuple = tuple(pred_entry["crumb_location_world_coords"])
                likelihood = pred_entry.get(f"evidence_likelihood_{agent_id}")
            
            if world_coord_tuple and likelihood is not None:
                detective_likelihoods[world_coord_tuple] = likelihood
        
        # Simulate detective crumb planting process
        num_simulations = 1000  # Same as detective path count (detective simulates 1000 paths)
        detective_plant_spots = []
        
        if detective_likelihoods:
            coords = list(detective_likelihoods.keys())
            likelihoods = list(detective_likelihoods.values())
            
            # Normalize to probabilities
            total_likelihood = sum(likelihoods)
            if total_likelihood > 0:
                probabilities = [lik / total_likelihood for lik in likelihoods]
                
                # Simulate detective's crumb planting decisions
                for _ in range(num_simulations):
                    chosen_coord = np.random.choice(len(coords), p=probabilities)
                    detective_plant_spots.append(coords[chosen_coord])
        
        detective_spots_counts = Counter(detective_plant_spots)
        
        # Create grids
        actual_grid = np.zeros((kitchen_height, kitchen_width))
        detective_grid = np.zeros((kitchen_height, kitchen_width))
        
        # Fill actual counts grid
        for world_coord_tuple, count in actual_spots_counts.items():
            kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
            if kitchen_coord:
                kx, ky = kitchen_coord
                if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                    actual_grid[ky, kx] = count
        
        # Fill detective simulated counts grid
        for world_coord_tuple, count in detective_spots_counts.items():
            kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
            if kitchen_coord:
                kx, ky = kitchen_coord
                if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                    detective_grid[ky, kx] = count
        
        # Create side-by-side comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 4))
        
        cmap_choice = "Reds" if agent_id == 'A' else "Blues"
        
        # Actual sophisticated planting
        sns.heatmap(actual_grid, ax=axes[0], annot=True, fmt=".0f", cmap=cmap_choice, cbar=True, linewidths=.5, linecolor='gray')
        axes[0].set_title(f"Agent {agent_id}: Actual Sophisticated Crumb Planting\n(Real counts from simulation)")
        axes[0].set_xlabel("Kitchen X")
        axes[0].set_ylabel("Kitchen Y")
        
        # Detective simulated counts
        sns.heatmap(detective_grid, ax=axes[1], annot=True, fmt=".0f", cmap=cmap_choice, cbar=True, linewidths=.5, linecolor='gray')
        axes[1].set_title(f"Agent {agent_id}: Detective Simulated Counts ({detective_agent_type.capitalize()})\n(Predicted counts from {num_simulations} simulations)")
        axes[1].set_xlabel("Kitchen X")
        axes[1].set_ylabel("Kitchen Y")
        
        # Set ticks for both subplots
        for ax in axes:
            ax.set_xticks(ticks=np.arange(0.5, kitchen_width, 1))
            ax.set_xticklabels(labels=np.arange(1, kitchen_width + 1))
            ax.set_yticks(ticks=np.arange(0.5, kitchen_height, 1))
            ax.set_yticklabels(labels=np.arange(kitchen_height, 0, -1))
        
        plt.suptitle(f"Crumb Planting Comparison - Agent {agent_id}, Trial: {trial_name}", fontsize=16)
        plt.tight_layout()
        
        # Save comparison plot
        comparison_filename = os.path.join(param_log_dir, f"crumb_planting_comparison_agent_{agent_id}_{detective_agent_type}_{trial_name}.png")
        try:
            plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved crumb planting comparison for Agent {agent_id} to {comparison_filename}")
        except Exception as e:
            logger.error(f"Error saving crumb planting comparison for Agent {agent_id}: {e}")
    plt.close()


def main(dir_to_search): 
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', stream=sys.stdout)

    agent_types_for_path_plots = ['naive', 'sophisticated', 'uniform']
    detective_model_types_for_plots = ['naive', 'sophisticated', 'uniform'] 

    processed_trials_for_detective_plots = set()

    for agent_type in agent_types_for_path_plots:
        search_pattern_csv = os.path.join(dir_to_search, '**', f'*_sampled_paths_{agent_type}.csv')
        csv_files = glob.glob(search_pattern_csv, recursive=True)

        for csv_file_path in csv_files:
            try:
                param_log_dir = os.path.dirname(csv_file_path)
                base_filename_csv = os.path.basename(csv_file_path)
                
                suffix_csv = f"_sampled_paths_{agent_type}.csv"
                trial_name = ""
                if base_filename_csv.endswith(suffix_csv):
                    trial_name = base_filename_csv[:-len(suffix_csv)]
                else:
                    logger.warning(f"Could not extract trial name from {base_filename_csv} using suffix {suffix_csv}. Skipping.")
                    continue
                
                logger.info(f"Processing Trial: '{trial_name}', Agent Type: '{agent_type}', Dir: '{param_log_dir}'")

                # Determine evidence type by trying to read metadata.json
                evidence_type = 'visual' # Default to visual
                metadata_path = os.path.join(param_log_dir, 'metadata.json') # Standard location
                # If not found, search one level up, as param_log_dir might be a sub-folder like w0.1_ntemp0.1...
                if not os.path.exists(metadata_path):
                    metadata_path_alt = os.path.join(os.path.dirname(param_log_dir), 'metadata.json')
                    if os.path.exists(metadata_path_alt):
                        metadata_path = metadata_path_alt
                    else: # check one more level up, for base trial dir
                        metadata_path_alt_2 = os.path.join(os.path.dirname(os.path.dirname(param_log_dir)), 'metadata.json')
                        if os.path.exists(metadata_path_alt_2):
                            metadata_path = metadata_path_alt_2
                        else:
                            logger.warning(f"metadata.json not found in {param_log_dir}, {os.path.dirname(param_log_dir)}, or {os.path.dirname(os.path.dirname(param_log_dir))}. Defaulting to visual evidence type.")

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        evidence_type = metadata.get('evidence', 'visual').lower()
                        logger.info(f"Determined evidence type: {evidence_type} from {metadata_path}")
                    except Exception as e:
                        logger.error(f"Error reading or parsing {metadata_path}: {e}. Defaulting to visual.")
                
                # Plot suspect paths heatmap
                if evidence_type == 'audio':
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, evidence_type='audio', path_segment_type='to_fridge')
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, evidence_type='audio', path_segment_type='return_from_fridge')
                
                elif evidence_type == 'visual':
                    # Plot both path segments for visual evidence, just like audio
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, evidence_type='visual', path_segment_type='to_fridge')
                    plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type, evidence_type='visual', path_segment_type='return_from_fridge')
                    if agent_type == 'sophisticated':
                        plot_suspect_crumb_planting_heatmap(trial_name, param_log_dir)
                
                # Detective prediction heatmaps (naive, sophisticated, and uniform)
                trial_context_key = (trial_name, param_log_dir)
                if trial_context_key not in processed_trials_for_detective_plots:
                    for detective_model_type in detective_model_types_for_plots:
                        # Updated to new naming convention: {trial_name}_{agent_type}_{evidence}_predictions.json
                        # For uniform, check both uniform_predictions.json and {trial_name}_uniform_{evidence}_predictions.json
                        json_pred_file = f"{trial_name}_{detective_model_type}_{evidence_type}_predictions.json"
                        json_pred_path = os.path.join(param_log_dir, json_pred_file)
                        
                        # Alternative filename for uniform (legacy format)
                        if detective_model_type == 'uniform' and not os.path.exists(json_pred_path):
                            json_pred_file_alt = f"{trial_name}_uniform_predictions.json"
                            json_pred_path_alt = os.path.join(param_log_dir, json_pred_file_alt)
                            if os.path.exists(json_pred_path_alt):
                                json_pred_path = json_pred_path_alt
                                json_pred_file = json_pred_file_alt
                        
                        if os.path.exists(json_pred_path):
                            logger.info(f"Plotting detective predictions for model '{detective_model_type}', trial '{trial_name}', evidence '{evidence_type}'.")
                            plot_detective_predictions_heatmap(trial_name, param_log_dir, detective_model_type, evidence_type)
                            
                            # Generate sophisticated crumb planting comparison (actual vs detective simulated)
                            if evidence_type == 'visual' and detective_model_type == 'sophisticated':
                                plot_crumb_planting_comparison(trial_name, param_log_dir, detective_model_type, evidence_type)
                    
                    # Generate retrospective smoothing comparison plots for visual evidence
                    if evidence_type == 'visual':
                        logger.info(f"Generating retrospective smoothing comparison for trial '{trial_name}'")
                        plot_retrospective_smoothing_comparison(trial_name, param_log_dir)
                    
                    processed_trials_for_detective_plots.add(trial_context_key)

            except Exception as e:
                logger.error(f"Error processing file {csv_file_path} for heatmap generation: {e}")
                import traceback
                traceback.print_exc() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for simulation results.")
    parser.add_argument("--dir", type=str, required=True, help="Results root directory to search for files and plot.")
    
    cli_args = parser.parse_args()
    
    main(cli_args.dir)
