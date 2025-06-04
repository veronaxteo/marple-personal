import logging
import pandas as pd
import ast
from src.core.world import World
from src.agents import Detective
from .base import BaseSimulator


class EmpiricalSimulator(BaseSimulator):
    """
    Simulator for empirical model based on human participant data.
    """
    def __init__(self, args, log_dir_base, param_log_dir, params, trials_to_run):
        super().__init__(args, log_dir_base, param_log_dir, params, trials_to_run)
        self.logger.info(f"Loading empirical paths from: {self.params.paths}")
        self.logger.info(f"Mismatched analysis: {self.params.mismatched}")
        
        if not self.params.paths:
            self.logger.error("Empirical path CSV file not specified")
            self.empirical_data = pd.DataFrame()
        else:
            self.empirical_data = self._load_empirical_data(self.params.paths)

    def _load_empirical_data(self, paths_csv: str) -> pd.DataFrame:
        """Load and validate empirical path data"""
        try:
            df = pd.read_csv(paths_csv)
            required_cols = ['trial', 'agent', 'agent_type', 'full_sequence_world_coords', 'middle_sequence_world_coords']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                exit()

            # Parse coordinate columns
            df['full_sequence_world_coords'] = df['full_sequence_world_coords'].apply(ast.literal_eval)
            df['middle_sequence_world_coords'] = df['middle_sequence_world_coords'].apply(ast.literal_eval)
            
            self.logger.info(f"Loaded {len(df)} empirical paths")
            return df
            
        except FileNotFoundError:
            self.logger.error(f"Empirical file not found: {paths_csv}")
            exit()
        except Exception as e:
            self.logger.error(f"Error loading empirical data: {e}")
            exit()

    def run_trial(self, trial_file: str, trial_name: str, world: World) -> dict:
        """Run empirical analysis for a single trial"""
        trial_data = self.empirical_data[self.empirical_data['trial'] == trial_name]

        if trial_data.empty:
            self.logger.warning(f"No empirical data found for trial {trial_name}")
            return None

        detective = Detective('detective_empirical', 'visual', self.params)

        # Organize path data by agent type
        path_data = self._organize_path_data(trial_data)
        
        # Run predictions for both naive and sophisticated
        results = {"trial": trial_name, "predictions": {}}
        
        for prediction_type in ['naive', 'sophisticated']:
            source_type = self._get_source_type(prediction_type)
            
            if not self._has_data_for_type(path_data, source_type):
                self.logger.warning(f"No {source_type} data for {prediction_type} prediction in {trial_name}")
                results["predictions"][prediction_type] = {}
                continue

            self.logger.info(f"Computing {prediction_type} predictions using {source_type} data")
            
            # TODO: Complete empirical implementation based on original simulation.py
        
        return results
            
    def _organize_path_data(self, trial_data):
        """Organize path data by agent type"""
        return {
            'naive': {'A': [], 'B': []},
            'sophisticated': {'A': [], 'B': []}
        }
    
    def _get_source_type(self, prediction_type):
        """Get source data type for predictions"""
        return 'sophisticated' if self.params.mismatched else prediction_type
    
    def _has_data_for_type(self, path_data, data_type):
        """Check if data exists for given type"""
        return (data_type in path_data and 
                path_data[data_type]['A'] and 
                path_data[data_type]['B']) 