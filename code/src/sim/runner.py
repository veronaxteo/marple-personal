"""
Unified simulation runner for all simulation types.
Provides complete orchestration of simulations with logging, metadata, and visualization.
"""

import logging
import os
import sys
import datetime
import numpy as np
import random
import json
from dataclasses import asdict
from typing import Dict, Any, Optional, List
from ..cfg import SimulationConfig
from ..sim import RSMSimulator, EmpiricalSimulator, UniformSimulator
from ..utils.io_utils import create_param_dir, get_json_files
from ..analysis.plot import create_simulation_plots


class SimulationRunner:
    """Simulation runner class."""
    def __init__(self, log_dir: str = "results"):
        self.log_dir = os.path.abspath(log_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_rsm_experiment(self, 
                          evidence_type: str, 
                          trial_name: str,
                          config_overrides: Optional[Dict[str, Any]] = None,
                          experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Run RSM simulation experiment."""
        return self._run_simulation_experiment(
            sim_type='rsm',
            evidence_type=evidence_type,
            trial_name=trial_name,
            config_overrides=config_overrides,
            experiment_name=experiment_name
        )
    
    def run_empirical_experiment(self,
                               empirical_paths_file: str,
                               trial_name: str = 'all',
                               mismatched_analysis: bool = False,
                               config_overrides: Optional[Dict[str, Any]] = None,
                               experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Run empirical analysis experiment."""
        
        results = self._run_simulation_experiment(
            sim_type='empirical',
            evidence_type='visual',  # Empirical uses visual evidence
            trial_name=trial_name,
            experiment_name=experiment_name
        )
        
        # Add empirical-specific attributes to config
        if hasattr(self, '_current_config'):
            self._current_config.empirical_paths_file = empirical_paths_file
            self._current_config.mismatched_analysis = mismatched_analysis
        
        return results
    
    def run_uniform_experiment(self,
                             evidence_type: str,
                             trial_name: str,
                             max_steps: int,
                             config_overrides: Optional[Dict[str, Any]] = None,
                             experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Run uniform baseline experiment."""
        uniform_overrides = {
            'default_trial': trial_name,
            'evidence.evidence_type': evidence_type,
            'sampling.max_steps': max_steps,
        }
        if config_overrides:
            uniform_overrides.update(config_overrides)
        
        return self._run_simulation_experiment(
            sim_type='uniform',
            evidence_type=evidence_type,
            trial_name=trial_name,
            config_overrides=uniform_overrides,
            experiment_name=experiment_name
        )
    
    def _run_simulation_experiment(self,
                                 sim_type: str,
                                 evidence_type: str,
                                 trial_name: str,
                                 config_overrides: Optional[Dict[str, Any]] = None,
                                 experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Internal method to run any simulation type."""
        
        # Create config
        config = self._create_config(evidence_type, trial_name, config_overrides or {})
        self._current_config = config  # Store for potential modification
        
        # Set random seed
        self._set_random_seed(config.seed)
        
        # Setup experiment directory
        exp_name = experiment_name or f"{sim_type}_{evidence_type}_{trial_name}"
        param_log_dir = create_param_dir(
            self.log_dir, 
            trial_name,
            evidence_type=evidence_type,
            max_steps=config.sampling.max_steps,
            model_type=sim_type,
            cost_weight=config.sampling.cost_weight,
            naive_temp=config.sampling.naive_temp,
            soph_temp=config.sampling.sophisticated_temp
        )
        
        # Setup logging
        self._setup_logging(param_log_dir, exp_name)
        self.logger.info(f"Starting {sim_type.upper()} experiment: {exp_name}")
        self.logger.info(f"Evidence type: {evidence_type}")
        
        # Save metadata
        self._save_metadata(config, param_log_dir)
        
        # Get trials to run
        trials_to_run = get_json_files(trial_name)
        self.logger.info(f"Trials to run: {trials_to_run}")
        
        # Run simulation
        simulator = self._create_simulator(sim_type, config, self.log_dir, param_log_dir, trials_to_run)
        results = simulator.run()
        
        # Generate plots
        self._generate_plots(param_log_dir, trials_to_run, evidence_type)
        
        self.logger.info(f"{sim_type.upper()} experiment '{exp_name}' completed")
        return results
    
    def _create_config(self, evidence_type: str, trial_name: str, overrides: Dict[str, Any]) -> SimulationConfig:
        """Create configuration for simulation."""
        # Get config file path
        cfg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg')
        evidence_config_path = os.path.join(cfg_dir, f'{evidence_type}.yaml')
        
        # Fallback to default.yaml if evidence-specific config doesn't exist
        if not os.path.exists(evidence_config_path):
            evidence_config_path = os.path.join(cfg_dir, 'default.yaml')
        
        # Always set trial and evidence type
        overrides['default_trial'] = trial_name
        overrides['evidence.evidence_type'] = evidence_type
        
        # Create config using the simplified factory method
        config = SimulationConfig.from_params(evidence_config_path, **overrides)
        
        # Set log directory
        config.log_dir_base = self.log_dir
        
        return config
    
    def _create_simulator(self, sim_type: str, config: SimulationConfig, 
                         log_dir_base: str, param_log_dir: str, trials_to_run: List[str]):
        """Create appropriate simulator instance."""
        if sim_type == 'rsm':
            return RSMSimulator(config, log_dir_base, param_log_dir, trials_to_run)
        elif sim_type == 'empirical':
            return EmpiricalSimulator(config, log_dir_base, param_log_dir, trials_to_run)
        elif sim_type == 'uniform':
            return UniformSimulator(config, log_dir_base, param_log_dir, trials_to_run)
        else:
            raise ValueError(f"Unknown simulation type: {sim_type}")
    
    def _set_random_seed(self, seed: int):
        """Set random seed for reproducible results."""
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f"Random seed set to: {seed}")
    
    def _setup_logging(self, log_dir: str, experiment_name: str):
        """Setup logging for experiment."""
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "simulation.log")
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup new logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger.info(f"Logging setup for experiment: {experiment_name}")
    
    def _save_metadata(self, config: SimulationConfig, param_log_dir: str):
        """Save simulation configuration metadata to JSON file."""
        metadata_filepath = os.path.join(param_log_dir, 'metadata.json')
        
        # Convert config to dictionary
        cfg_dict = asdict(config)
        
        # Clean up parameters based on evidence type
        evidence_type = cfg_dict.get('evidence', {}).get('evidence_type')
        
        if 'evidence' in cfg_dict:
            evidence_dict = cfg_dict['evidence']
            
            # Always remove these large fields
            large_fields = [
                'naive_A_visual_likelihoods_map', 'naive_B_visual_likelihoods_map',
                'naive_A_to_fridge_steps_model', 'naive_A_from_fridge_steps_model',
                'naive_B_to_fridge_steps_model', 'naive_B_from_fridge_steps_model'
            ]
            for field in large_fields:
                evidence_dict.pop(field, None)
                
            if evidence_type == 'visual':
                # Remove audio params
                audio_params = ['audio_similarity_sigma', 'audio_gt_step_size']
                for param in audio_params:
                    evidence_dict.pop(param, None)
                    
            elif evidence_type == 'audio':
                # Remove visual params
                visual_params = [
                    'naive_detective_sigma', 'crumb_planting_sigma', 
                    'sophisticated_detective_sigma',
                    'visual_naive_likelihood_alpha', 'visual_sophisticated_likelihood_alpha'
                ]
                for param in visual_params:
                    evidence_dict.pop(param, None)
        
        # Add timestamp
        cfg_dict['simulation_timestamp'] = datetime.datetime.now().isoformat()
        
        with open(metadata_filepath, 'w') as f_meta:
            json.dump(cfg_dict, f_meta, indent=4)
        
        self.logger.info(f"Saved metadata to {metadata_filepath}")
    
    def _generate_plots(self, param_log_dir: str, trials_to_run: List[str], evidence_type: str):
        """Generate plots for experiment results."""
        self.logger.info("Generating plots...")
        
        for trial_name in trials_to_run:
            trial_name_clean = trial_name.replace('_A1.json', '').replace('.json', '')
            try:
                create_simulation_plots(param_log_dir, trial_name_clean, evidence_type)
                self.logger.info(f"Generated {evidence_type} plots for {trial_name_clean}")
            except Exception as e:
                self.logger.warning(f"Could not generate plots for {trial_name_clean}: {e}")
    
    # TODO: implement batch experiments
    def run_batch_experiments(self, experiment_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run multiple experiments in batch.
        
        Each experiment spec should have:
        - sim_type: 'rsm', 'empirical', or 'uniform'  
        - evidence_type: str
        - trial_name: str
        - config_overrides: dict (optional)
        - experiment_name: str (optional)
        - Additional type-specific parameters
        """
        all_results = {}
        
        for i, spec in enumerate(experiment_specs):
            exp_name = spec.get('experiment_name', f"experiment_{i+1}")
            sim_type = spec.get('sim_type', 'rsm')
            
            self.logger.info(f"Starting batch experiment {i+1}/{len(experiment_specs)}: {exp_name}")
            
            try:
                if sim_type == 'rsm':
                    results = self.run_rsm_experiment(
                        evidence_type=spec['evidence_type'],
                        trial_name=spec['trial_name'],
                        config_overrides=spec.get('config_overrides'),
                        experiment_name=exp_name
                    )
                elif sim_type == 'empirical':
                    results = self.run_empirical_experiment(
                        empirical_paths_file=spec['empirical_paths_file'],
                        trial_name=spec.get('trial_name', 'all'),
                        mismatched_analysis=spec.get('mismatched_analysis', False),
                        config_overrides=spec.get('config_overrides'),
                        experiment_name=exp_name
                    )
                elif sim_type == 'uniform':
                    results = self.run_uniform_experiment(
                        evidence_type=spec['evidence_type'],
                        trial_name=spec['trial_name'],
                        max_steps=spec.get('max_steps', 25),
                        config_overrides=spec.get('config_overrides'),
                        experiment_name=exp_name
                    )
                else:
                    raise ValueError(f"Unknown simulation type: {sim_type}")
                
                all_results[exp_name] = results
                
            except Exception as e:
                self.logger.error(f"Experiment {exp_name} failed: {e}")
                all_results[exp_name] = {"error": str(e)}
        
        return all_results 
