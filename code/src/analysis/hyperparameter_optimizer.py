"""
Hyperparameter optimization for visual evidence case.

Optimizes parameters:
1. naive_temp 
2. sophisticated_temp
3. cost_weight
4. naive_detective_sigma
5. sophisticated_detective_sigma
6. crumb_planting_sigma
7. visual_naive_likelihood_alpha
8. visual_sophisticated_likelihood_alpha

Uses multi-objective Bayesian optimization to optimize:
- Detective predictions: RMSE, Pearson correlation, AIC
- Suspect paths: Earth Mover's Distance (EMD)
"""

import os
import logging
import numpy as np
import pandas as pd
import optuna
from scipy.stats import pearsonr
from scipy.spatial.distance import wasserstein_distance
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Any
import ast

from src.cfg import SimulationConfig
from src.sim import RSMSimulator
from src.utils.io_utils import get_json_files, create_param_dir


class HyperparameterOptimizer:
    """Hyperparameter optimizer for visual evidence case"""
    
    def __init__(self, 
                 human_trials_csv: str, 
                 human_paths_csv: str,
                 base_config_path: str = 'visual.yaml',
                 log_dir: str = 'results/optimization',
                 n_trials: int = 100):
        """
        Initialize optimizer.
        
        Args:
            human_trials_csv: Path to human detective prediction data
            human_paths_csv: Path to human suspect path data  
            base_config_path: Base YAML config to use
            log_dir: Directory for optimization logs
            n_trials: Number of optimization trials
        """
        self.human_trials_csv = human_trials_csv
        self.human_paths_csv = human_paths_csv
        self.base_config_path = base_config_path
        self.log_dir = os.path.abspath(log_dir)
        self.n_trials = n_trials
        
        # Load human data
        self.human_trials = pd.read_csv(human_trials_csv)
        self.human_paths = pd.read_csv(human_paths_csv)
        
        # Setup logging
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'optimization.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'naive_temp': (0.01, 1.0),  # log scale
            'sophisticated_temp': (0.01, 1.0),  # log scale  
            'cost_weight': (1.0, 50.0),
            'naive_detective_sigma': (0.1, 3.0),
            'sophisticated_detective_sigma': (0.5, 5.0),
            'crumb_planting_sigma': (0.1, 5.0),
            'visual_naive_likelihood_alpha': (0.1, 5.0),
            'visual_sophisticated_likelihood_alpha': (0.1, 5.0),
        }
        
        self.logger.info(f"Initialized optimizer with {len(self.human_trials)} human trials")
        self.logger.info(f"Parameter ranges: {self.param_ranges}")
    
    def objective(self, trial: optuna.Trial) -> Tuple[float, float, float, float]:
        """
        Objective function for multi-objective optimization.
        
        Returns:
            Tuple of (detective_rmse, -detective_correlation, detective_aic, suspect_emd)
            All objectives are minimized (correlation is negated)
        """
        # Sample hyperparameters
        params = {
            'naive_temp': trial.suggest_float('naive_temp', *self.param_ranges['naive_temp'], log=True),
            'sophisticated_temp': trial.suggest_float('sophisticated_temp', *self.param_ranges['sophisticated_temp'], log=True),
            'cost_weight': trial.suggest_float('cost_weight', *self.param_ranges['cost_weight']),
            'naive_detective_sigma': trial.suggest_float('naive_detective_sigma', *self.param_ranges['naive_detective_sigma']),
            'sophisticated_detective_sigma': trial.suggest_float('sophisticated_detective_sigma', *self.param_ranges['sophisticated_detective_sigma']),
            'crumb_planting_sigma': trial.suggest_float('crumb_planting_sigma', *self.param_ranges['crumb_planting_sigma']),
            'visual_naive_likelihood_alpha': trial.suggest_float('visual_naive_likelihood_alpha', *self.param_ranges['visual_naive_likelihood_alpha']),
            'visual_sophisticated_likelihood_alpha': trial.suggest_float('visual_sophisticated_likelihood_alpha', *self.param_ranges['visual_sophisticated_likelihood_alpha']),
        }
        
        try:
            # Run simulation with these parameters
            detective_predictions, suspect_paths = self._run_simulation(params, trial.number)
            
            # Calculate metrics
            detective_metrics = self._calculate_detective_metrics(detective_predictions)
            suspect_metrics = self._calculate_suspect_metrics(suspect_paths)
            
            self.logger.info(f"Trial {trial.number}: RMSE={detective_metrics['rmse']:.3f}, "
                           f"Corr={detective_metrics['correlation']:.3f}, "
                           f"AIC={detective_metrics['aic']:.1f}, EMD={suspect_metrics['emd']:.3f}")
            
            return (
                detective_metrics['rmse'],
                -detective_metrics['correlation'],  # Minimize negative correlation
                detective_metrics['aic'], 
                suspect_metrics['emd']
            )
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            # Return poor scores for failed trials
            return (100.0, 1.0, 10000.0, 100.0)
    
    def _run_simulation(self, params: Dict[str, float], trial_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run simulation with given parameters"""
        
        # Create trial directory
        trial_dir = os.path.join(self.log_dir, f"trial_{trial_number:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Get all available trials
        trials_to_run = self._get_available_trials()
        
        all_detective_predictions = []
        all_suspect_paths = []
        
        for trial_name in trials_to_run:
            try:
                # Create config for this trial
                config = SimulationConfig.for_optimization(
                    self.base_config_path,
                    trial_name=trial_name,
                    **params
                )
                
                # Setup trial-specific logging directory
                param_log_dir = os.path.join(trial_dir, trial_name)
                os.makedirs(param_log_dir, exist_ok=True)
                
                # Run simulation
                simulator = RSMSimulator(config, self.log_dir, param_log_dir, [trial_name])
                results = simulator.run()
                
                # Collect results
                detective_pred_file = os.path.join(param_log_dir, f"{trial_name}_detective_predictions.csv")
                suspect_path_file = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_naive.csv")
                suspect_path_file_soph = os.path.join(param_log_dir, f"{trial_name}_sampled_paths_sophisticated.csv")
                
                if os.path.exists(detective_pred_file):
                    detective_df = pd.read_csv(detective_pred_file)
                    detective_df['trial'] = trial_name
                    all_detective_predictions.append(detective_df)
                
                for agent_type, file_path in [('naive', suspect_path_file), ('sophisticated', suspect_path_file_soph)]:
                    if os.path.exists(file_path):
                        paths_df = pd.read_csv(file_path)
                        paths_df['trial'] = trial_name
                        paths_df['agent_type'] = agent_type
                        all_suspect_paths.append(paths_df)
                        
            except Exception as e:
                self.logger.warning(f"Failed to run trial {trial_name}: {e}")
                continue
        
        # Combine results
        detective_predictions = pd.concat(all_detective_predictions, ignore_index=True) if all_detective_predictions else pd.DataFrame()
        suspect_paths = pd.concat(all_suspect_paths, ignore_index=True) if all_suspect_paths else pd.DataFrame()
        
        return detective_predictions, suspect_paths
    
    def _get_available_trials(self) -> List[str]:
        """Get list of available trial names from human data"""
        # Use a subset of trials for optimization to speed up
        available_trials = self.human_trials['trial'].unique()
        # For optimization, use a smaller subset
        return list(available_trials)[:5]  # Use first 5 trials for speed
    
    def _calculate_detective_metrics(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Calculate detective prediction metrics"""
        if predictions.empty:
            return {'rmse': 100.0, 'correlation': 0.0, 'aic': 10000.0}
        
        # Match predictions to human data
        matched_pred, matched_human = self._match_detective_predictions(predictions)
        
        if len(matched_pred) == 0:
            return {'rmse': 100.0, 'correlation': 0.0, 'aic': 10000.0}
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(matched_human, matched_pred))
        
        # Pearson correlation
        correlation, _ = pearsonr(matched_human, matched_pred)
        if np.isnan(correlation):
            correlation = 0.0
        
        # AIC (assuming Gaussian likelihood)
        n = len(matched_human)
        residuals = np.array(matched_human) - np.array(matched_pred)
        sigma_squared = np.var(residuals)
        if sigma_squared > 0:
            log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)
            k = len(self.param_ranges)  # number of parameters
            aic = 2 * k - 2 * log_likelihood
        else:
            aic = 10000.0
        
        return {
            'rmse': rmse,
            'correlation': correlation,
            'aic': aic
        }
    
    def _match_detective_predictions(self, predictions: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Match model predictions to human data"""
        matched_pred = []
        matched_human = []
        
        for _, human_row in self.human_trials.iterrows():
            # Find matching model prediction
            model_match = predictions[
                (predictions['trial'] == human_row['trial']) &
                (predictions['agent_type'] == human_row['agent_type']) &
                (predictions['id'] == human_row['id'])
            ]
            
            if not model_match.empty:
                matched_pred.append(model_match.iloc[0]['response'])
                matched_human.append(human_row['response'])
        
        return matched_pred, matched_human
    
    def _calculate_suspect_metrics(self, paths: pd.DataFrame) -> Dict[str, float]:
        """Calculate suspect path metrics using Earth Mover's Distance"""
        if paths.empty:
            return {'emd': 100.0}
        
        emd_scores = []
        
        for trial in self.human_paths['trial'].unique():
            if trial not in paths['trial'].values:
                continue
                
            for agent_type in ['naive', 'sophisticated']:
                try:
                    human_trial_paths = self._extract_paths_for_trial(self.human_paths, trial, agent_type)
                    model_trial_paths = self._extract_paths_for_trial(paths, trial, agent_type)
                    
                    if len(human_trial_paths) > 0 and len(model_trial_paths) > 0:
                        emd = self._calculate_path_emd(human_trial_paths, model_trial_paths)
                        emd_scores.append(emd)
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate EMD for {trial} {agent_type}: {e}")
                    continue
        
        return {
            'emd': np.mean(emd_scores) if emd_scores else 100.0
        }
    
    def _extract_paths_for_trial(self, paths_df: pd.DataFrame, trial: str, agent_type: str) -> List[List[Tuple[int, int]]]:
        """Extract coordinate paths for a specific trial and agent type"""
        trial_data = paths_df[
            (paths_df['trial'] == trial) & 
            (paths_df['agent_type'] == agent_type)
        ]
        
        extracted_paths = []
        for _, row in trial_data.iterrows():
            try:
                if 'middle_sequence_world_coords' in row:
                    path_coords = ast.literal_eval(row['middle_sequence_world_coords'])
                elif 'middle_sequence' in row:
                    path_coords = ast.literal_eval(row['middle_sequence'])
                else:
                    continue
                    
                extracted_paths.append(path_coords)
            except Exception:
                continue
                
        return extracted_paths
    
    def _calculate_path_emd(self, human_paths: List[List[Tuple[int, int]]], 
                           model_paths: List[List[Tuple[int, int]]]) -> float:
        """Calculate Earth Mover's Distance between path distributions"""
        # Simplified EMD calculation - convert paths to summary statistics
        # You might want to implement a more sophisticated distance measure
        
        def path_to_features(paths):
            features = []
            for path in paths:
                if len(path) > 0:
                    features.append([
                        len(path),  # path length
                        path[0][0], path[0][1],  # start coordinates
                        path[-1][0], path[-1][1],  # end coordinates
                    ])
            return np.array(features) if features else np.array([]).reshape(0, 5)
        
        human_features = path_to_features(human_paths)
        model_features = path_to_features(model_paths)
        
        if len(human_features) == 0 or len(model_features) == 0:
            return 100.0
        
        # Calculate EMD for each feature dimension and average
        emd_scores = []
        for i in range(human_features.shape[1]):
            try:
                emd = wasserstein_distance(human_features[:, i], model_features[:, i])
                emd_scores.append(emd)
            except Exception:
                emd_scores.append(100.0)
        
        return np.mean(emd_scores)
    
    def optimize(self) -> optuna.Study:
        """Run multi-objective optimization"""
        self.logger.info(f"Starting optimization with {self.n_trials} trials")
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['minimize', 'minimize', 'minimize', 'minimize'],
            sampler=optuna.samplers.MOTPESampler(seed=42),
            study_name='hyperparameter_optimization'
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Log results
        self.logger.info(f"Optimization completed. {len(study.trials)} trials run")
        self.logger.info(f"Number of Pareto optimal solutions: {len(study.best_trials)}")
        
        # Save results
        results_file = os.path.join(self.log_dir, 'optimization_results.csv')
        self._save_results(study, results_file)
        
        return study
    
    def _save_results(self, study: optuna.Study, results_file: str):
        """Save optimization results to CSV"""
        results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {
                    'trial_number': trial.number,
                    'detective_rmse': trial.values[0],
                    'detective_correlation': -trial.values[1],  # Convert back to positive
                    'detective_aic': trial.values[2],
                    'suspect_emd': trial.values[3],
                    **trial.params
                }
                results.append(row)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Results saved to {results_file}")


def main():
    """Main entry point for optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for visual evidence case")
    parser.add_argument('--human-trials', required=True, help='Path to human detective trials CSV')
    parser.add_argument('--human-paths', required=True, help='Path to human suspect paths CSV')
    parser.add_argument('--config', default='visual.yaml', help='Base config file')
    parser.add_argument('--log-dir', default='results/optimization', help='Log directory')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    
    args = parser.parse_args()
    
    optimizer = HyperparameterOptimizer(
        human_trials_csv=args.human_trials,
        human_paths_csv=args.human_paths,
        base_config_path=args.config,
        log_dir=args.log_dir,
        n_trials=args.n_trials
    )
    
    study = optimizer.optimize()
    
    print(f"\nOptimization completed!")
    print(f"Best trials found: {len(study.best_trials)}")
    print(f"Results saved to: {args.log_dir}")


if __name__ == "__main__":
    main() 