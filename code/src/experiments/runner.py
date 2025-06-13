"""
Experiment runner for coordinating multiple simulations.

Provides clean orchestration of RSM simulations with different evidence types.
"""

import logging
import os
from typing import Dict, Any, Optional

from ..cfg import SimulationConfig
from ..sim import RSMSimulator
from ..utils.io_utils import create_param_dir, get_json_files
from ..analysis.plot import create_simulation_plots


class ExperimentRunner:
    """Experiment runner that uses YAML configs"""
    
    def __init__(self, log_dir: str = "results"):
        self.log_dir = os.path.abspath(log_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_rsm_experiment(self, 
                          evidence_type: str, 
                          trial_name: str,
                          config_overrides: Optional[Dict[str, Any]] = None,
                          experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run an RSM experiment using evidence-specific YAML configs.
        
        Args:
            evidence_type: 'visual', 'audio', or 'multimodal'
            trial_name: Trial to run (e.g., 'snack1')
            config_overrides: Optional dict to override any config values
            experiment_name: Optional name for the experiment (for logging)
        """
        
        # Load config using the evidence-specific YAML
        config = SimulationConfig.from_evidence_type(
            evidence_type=evidence_type,
            trial_name=trial_name,
            **(config_overrides or {})
        )
        
        # Setup experiment directory
        exp_name = experiment_name or f"{evidence_type}_{trial_name}"
        param_log_dir = create_param_dir(
            self.log_dir, 
            trial_name,
            evidence_type=evidence_type,
            max_steps=config.sampling.max_steps,
            model_type="rsm",
            cost_weight=config.sampling.cost_weight,
            naive_temp=config.sampling.naive_temp,
            soph_temp=config.sampling.sophisticated_temp
        )
        
        # Setup logging
        self._setup_logging(param_log_dir, exp_name)
        self.logger.info(f"Starting RSM experiment: {exp_name}")
        self.logger.info(f"Evidence type: {evidence_type}")
        self.logger.info(f"Config loaded from: {evidence_type}.yaml")
        
        # Get trials to run
        trials_to_run = get_json_files(trial_name)
        
        # Run simulation
        simulator = RSMSimulator(config, self.log_dir, param_log_dir, trials_to_run)
        results = simulator.run()
        
        # Generate plots
        self._generate_plots(param_log_dir, trials_to_run, evidence_type)
        
        self.logger.info(f"RSM experiment '{exp_name}' completed")
        return results
    
    # TODO: add empirical and uniform runners
    
    def run_batch_experiments(self, experiment_specs: list) -> Dict[str, Any]:
        """
        Run multiple experiments in batch.
        
        Args:
            experiment_specs: List of dicts with keys:
                - evidence_type: str
                - trial_name: str  
                - config_overrides: dict (optional)
                - experiment_name: str (optional)
        """
        all_results = {}
        
        for i, spec in enumerate(experiment_specs):
            exp_name = spec.get('experiment_name', f"experiment_{i+1}")
            self.logger.info(f"Starting batch experiment {i+1} / {len(experiment_specs)}: {exp_name}")
            
            try:
                results = self.run_rsm_experiment(
                    evidence_type=spec['evidence_type'],
                    trial_name=spec['trial_name'],
                    config_overrides=spec.get('config_overrides'),
                    experiment_name=exp_name
                )
                all_results[exp_name] = results
                
            except Exception as e:
                self.logger.error(f"Experiment {exp_name} failed: {e}")
                all_results[exp_name] = {"error": str(e)}
        
        return all_results
    
    def _setup_logging(self, log_dir: str, experiment_name: str):
        """Setup logging for experiment"""
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
    
    def _generate_plots(self, param_log_dir: str, trials_to_run: list, evidence_type: str):
        """Generate plots for experiment results"""
        self.logger.info("Generating plots...")
        
        for trial_name in trials_to_run:
            trial_name_clean = trial_name.replace('_A1.json', '').replace('.json', '')
            try:
                create_simulation_plots(param_log_dir, trial_name_clean, evidence_type)
                self.logger.info(f"Generated {evidence_type} plots for {trial_name_clean}")
            except Exception as e:
                self.logger.warning(f"Could not generate plots for {trial_name_clean}: {e}") 
