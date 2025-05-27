import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import logging
from world import World
import ast
import argparse
import sys
import glob
import json


def plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type_to_plot):
    """
    Generates and saves heatmaps of sampled path frequencies for suspects A and B.
    """
    logger = logging.getLogger(__name__)

    trial_file_name = f"{trial_name}_A1.json"
    world = World.initialize_world_start(trial_file_name)
    kitchen_width = world.kitchen_width
    kitchen_height = world.kitchen_height

    csv_file_path = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_{agent_type_to_plot}.csv")
    paths_df = pd.read_csv(csv_file_path)

    path_column_name = 'full_sequence_world_coords'
    fridge_access_point = world.get_fridge_access_point()

    for agent_id in ['A', 'B']:
        agent_df = paths_df[paths_df['agent'] == agent_id]

        agent_paths_str_list = agent_df[path_column_name].tolist()
        heatmap_grid = np.zeros((kitchen_height, kitchen_width))
        paths_processed_count = 0

        for path_str in agent_paths_str_list:
            path_coords_world_full = ast.literal_eval(path_str)
            
            path_coords_to_plot = []
            if fridge_access_point and path_coords_world_full:
                fridge_ap_tuple = tuple(fridge_access_point)
                # Convert all path coords to tuples of ints for consistent matching
                path_coords_world_tuples = [tuple(map(int, coord)) for coord in path_coords_world_full if isinstance(coord, (list, tuple)) and len(coord) == 2]

                fridge_idx = -1
                for i, coord_tuple in enumerate(path_coords_world_tuples):
                    if coord_tuple == fridge_ap_tuple:
                        fridge_idx = i
                        break
                
                if fridge_idx != -1:
                    path_coords_to_plot = path_coords_world_tuples[fridge_idx:] # From fridge onwards
                else:
                    pass 
                
            elif not fridge_access_point: # If no fridge_access_point defined for the world, plot full path as fallback.
                path_coords_to_plot = [tuple(map(int, coord)) for coord in path_coords_world_full if isinstance(coord, (list, tuple)) and len(coord) == 2]

            paths_processed_count +=1
            for world_coord_tuple_anytype in path_coords_to_plot: 
                if isinstance(world_coord_tuple_anytype, (list, tuple)) and len(world_coord_tuple_anytype) == 2 and \
                   all(isinstance(c, (int, float)) for c in world_coord_tuple_anytype):
                    world_coord_tuple = tuple(map(int, world_coord_tuple_anytype))
                    kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
                    if kitchen_coord:
                        kx, ky = kitchen_coord
                        if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                            heatmap_grid[ky, kx] += 1

        plt.figure(figsize=(10, 4))
        
        # Set color based on agent_id
        plot_cmap = "Blues" if agent_id == 'A' else "Greens"
        
        sns.heatmap(heatmap_grid, annot=True, fmt=".0f", cmap=plot_cmap, cbar=True, linewidths=.5, linecolor='gray')
        plt.title(f"Agent {agent_id} ({agent_type_to_plot.capitalize()}): Return Path Tile Counts \n Trial: {trial_name}")
        plt.xlabel("Kitchen X")
        plt.ylabel("Kitchen Y")
        
        plt.xticks(ticks=np.arange(0.5, kitchen_width, 1), labels=np.arange(1, kitchen_width + 1))
        plt.yticks(ticks=np.arange(0.5, kitchen_height, 1), labels=np.arange(kitchen_height, 0, -1))

        heatmap_filename = os.path.join(param_log_dir, f"sampled_paths_heatmap_agent_{agent_id}_{agent_type_to_plot}_{trial_name}.png")
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

    kitchen_width = world.kitchen_width
    kitchen_height = world.kitchen_height

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


def plot_detective_predictions_heatmap(trial_name, param_log_dir, detective_agent_type):
    """
    Generates and saves a heatmap of detective slider predictions for each crumb location.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    trial_file_name = f"{trial_name}_A1.json" 
    try:
        world = World.initialize_world_start(trial_file_name)
        logger.info(f"Successfully loaded world for trial: {trial_name} to plot detective predictions.")
    except Exception as e:
        logger.error(f"Failed to load world for trial {trial_name} using {trial_file_name}: {e}. Cannot generate detective prediction heatmap.")
        return

    kitchen_width = world.kitchen_width
    kitchen_height = world.kitchen_height

    json_file_path = os.path.join(param_log_dir, f"detective_preds_{trial_name}_{detective_agent_type}.json")
    if not os.path.exists(json_file_path):
        logger.error(f"Detective predictions JSON file not found: {json_file_path}. Cannot generate heatmap.")
        return

    try:
        with open(json_file_path, 'r') as f:
            predictions_data = json.load(f)
        logger.info(f"Loaded detective predictions from {json_file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {json_file_path}: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading predictions file {json_file_path}: {e}")
        return

    if not predictions_data:
        logger.info(f"No prediction data found in {json_file_path} for {detective_agent_type} detective. Skipping heatmap.")
        return

    heatmap_grid = np.full((kitchen_height, kitchen_width), np.nan) # Initialize with NaN for missing spots

    for pred_entry in predictions_data:
        world_coord_tuple = tuple(pred_entry.get("crumb_location_world_coords", []))
        slider_prediction = pred_entry.get("slider_prediction")

        if not world_coord_tuple or slider_prediction is None:
            logger.warning(f"Skipping entry with missing data: {pred_entry}")
            continue
        
        try:
            kitchen_coord = world.world_to_kitchen_coords(world_coord_tuple[0], world_coord_tuple[1])
            kx, ky = kitchen_coord # kx = col, ky = row
            if 0 <= ky < kitchen_height and 0 <= kx < kitchen_width:
                heatmap_grid[ky, kx] = slider_prediction
            else:
                logger.warning(f"Kitchen coordinates {kitchen_coord} out of bounds for world coord {world_coord_tuple}.")
        except Exception as e:
            logger.error(f"Error converting/placing world coord {world_coord_tuple} to kitchen grid: {e}")


    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_grid, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True, vmin=-50, vmax=50,  linewidths=.5, linecolor='gray')
    plt.title(f"Detective Predictions ({detective_agent_type.capitalize()})\nTrial: {trial_name}, Slider: (-50: A, 50: B)")
    plt.xlabel("Kitchen X")
    plt.ylabel("Kitchen Y")
    
    plt.xticks(ticks=np.arange(0.5, kitchen_width, 1), labels=np.arange(1, kitchen_width + 1))
    plt.yticks(ticks=np.arange(0.5, kitchen_height, 1), labels=np.arange(kitchen_height, 0, -1)) # Flipped y-axis

    heatmap_filename = os.path.join(param_log_dir, f"detective_preds_heatmap_{detective_agent_type}_{trial_name}.png")
    try:
        plt.savefig(heatmap_filename)
        logger.info(f"Saved detective predictions heatmap to {heatmap_filename}")
    except Exception as e:
        logger.error(f"Error saving detective predictions heatmap: {e}")
    plt.close()


def main(dir_to_search): 
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', stream=sys.stdout)

    agent_types_for_path_plots = ['naive', 'sophisticated']
    detective_model_types_for_plots = ['naive', 'sophisticated'] 

    processed_trials_for_detective_plots = set()

    for agent_type in agent_types_for_path_plots:
        search_pattern_csv = os.path.join(dir_to_search, '**', f'*_sampled_paths_{agent_type}.csv')
        csv_files = glob.glob(search_pattern_csv, recursive=True)

        if not csv_files:
            logger.info(f"No sampled path CSVs found for agent type '{agent_type}' in '{dir_to_search}'.")

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

                # Plot suspect paths heatmap
                plot_suspect_paths_heatmap(trial_name, param_log_dir, agent_type)

                if agent_type == 'sophisticated':
                    plot_suspect_crumb_planting_heatmap(trial_name, param_log_dir)
                
                # Detective prediction heatmaps (naive and sophisticated models)
                trial_context_key = (trial_name, param_log_dir)
                if trial_context_key not in processed_trials_for_detective_plots:
                    for detective_model_type in detective_model_types_for_plots:
                        json_pred_file = f"detective_preds_{trial_name}_{detective_model_type}.json"
                        json_pred_path = os.path.join(param_log_dir, json_pred_file)
                        
                        if os.path.exists(json_pred_path):
                            logger.info(f"Plotting detective predictions for model '{detective_model_type}', trial '{trial_name}'.")
                            plot_detective_predictions_heatmap(trial_name, param_log_dir, detective_model_type)
                    
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
