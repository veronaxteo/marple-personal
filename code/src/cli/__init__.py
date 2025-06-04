"""
Command Line Interface for simulation.

Provides command-line interface for running different simulation types
including RSM, empirical analysis, and uniform baseline models.
"""

import argparse
import logging
import sys
import os
import datetime
import numpy as np
import random
import json
import yaml
from dataclasses import asdict

from ..utils.io_utils import get_json_files, create_param_dir
from ..cfg import SimulationConfig, SamplingConfig, EvidenceConfig
from ..sim import RSMSimulator, EmpiricalSimulator, UniformSimulator
from ..analysis.plotting import create_simulation_plots


def setup_logging(log_dir, log_file=None):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"simulation_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_path


def set_random_seed(seed: int):
    """Set random seed for reproducible results"""
    np.random.seed(seed)
    random.seed(seed)
    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to: {seed}")


def save_metadata(config, param_log_dir):
    """Save simulation configuration metadata to JSON file"""
    metadata_filepath = os.path.join(param_log_dir, 'metadata.json')
    
    # Convert config to dictionary
    cfg_dict_for_metadata = asdict(config)
    
    # Remove large fields that would make the file unnecessarily big
    cfg_dict_for_metadata.pop('naive_A_visual_likelihoods_map', None)
    cfg_dict_for_metadata.pop('naive_B_visual_likelihoods_map', None)
    cfg_dict_for_metadata.pop('naive_A_to_fridge_steps_model', None)
    cfg_dict_for_metadata.pop('naive_A_from_fridge_steps_model', None)
    cfg_dict_for_metadata.pop('naive_B_to_fridge_steps_model', None)
    cfg_dict_for_metadata.pop('naive_B_from_fridge_steps_model', None)
    
    # Remove any other large data structures if they exist
    if 'evidence' in cfg_dict_for_metadata:
        evidence_dict = cfg_dict_for_metadata['evidence']
        evidence_dict.pop('naive_A_visual_likelihoods_map', None)
        evidence_dict.pop('naive_B_visual_likelihoods_map', None)
        evidence_dict.pop('naive_A_to_fridge_steps_model', None)
        evidence_dict.pop('naive_A_from_fridge_steps_model', None)
        evidence_dict.pop('naive_B_to_fridge_steps_model', None)
        evidence_dict.pop('naive_B_from_fridge_steps_model', None)
    
    # Add timestamp
    cfg_dict_for_metadata['simulation_timestamp'] = datetime.datetime.now().isoformat()
    
    with open(metadata_filepath, 'w') as f_meta:
        json.dump(cfg_dict_for_metadata, f_meta, indent=4)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saved metadata to {metadata_filepath}")


def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s rsm --evidence visual --trial snack1 --max-steps 25
        %(prog)s empirical --paths results/paths.csv --mismatched
        %(prog)s uniform --trial snack2
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Simulation type')
    
    # RSM simulation
    rsm_parser = subparsers.add_parser('rsm', help='Run RSM simulation')
    rsm_parser.add_argument('--evidence', choices=['visual', 'audio', 'multimodal'], 
                           default='visual', help='Evidence type')
    rsm_parser.add_argument('--trial', default='snack1', help='Trial name (or "all")')
    rsm_parser.add_argument('--max-steps', type=int, default=25, help='Maximum steps')
    rsm_parser.add_argument('--max-steps-middle', type=int, default=0, help='Max middle steps')
    rsm_parser.add_argument('--weight', type=float, default=0.1, help='Cost weight')
    rsm_parser.add_argument('--naive-temp', type=float, default=0.01, help='Naive temperature')
    rsm_parser.add_argument('--soph-temp', type=float, default=0.01, help='Sophisticated temperature')
    rsm_parser.add_argument('--log-dir', default='../../results', help='Log directory')
    
    # Empirical analysis  
    emp_parser = subparsers.add_parser('empirical', help='Run empirical analysis')
    emp_parser.add_argument('--paths', required=True, help='CSV file with empirical paths')
    emp_parser.add_argument('--trial', default='all', help='Trial name (or "all")')
    emp_parser.add_argument('--mismatched', action='store_true', help='Mismatched analysis')
    emp_parser.add_argument('--log-dir', default='../../results', help='Log directory')
    
    # Uniform baseline
    uniform_parser = subparsers.add_parser('uniform', help='Run uniform baseline')
    uniform_parser.add_argument('--trial', default='snack1', help='Trial name (or "all")')
    uniform_parser.add_argument('--max-steps', type=int, default=25, help='Maximum steps')
    uniform_parser.add_argument('--log-dir', default='../../results', help='Log directory')
    
    return parser


def create_config_from_args(args):
    """Create SimulationConfig object from CLI arguments"""
    # Load base configuration from YAML defaults
    defaults_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg', 'default.yaml')
    
    if args.command == 'rsm':
        # Load defaults from YAML
        with open(defaults_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        # Update defaults with CLI arguments
        defaults['default_trial'] = args.trial
        defaults['sampling']['cost_weight'] = args.weight
        defaults['sampling']['naive_temp'] = args.naive_temp
        defaults['sampling']['sophisticated_temp'] = args.soph_temp
        defaults['sampling']['max_steps'] = args.max_steps
        defaults['evidence']['evidence_type'] = args.evidence
        
        # Add evidence-specific parameters
        if args.evidence == 'audio':
            defaults['evidence']['audio_similarity_sigma'] = 0.1
        elif args.evidence == 'multimodal':
            defaults['evidence']['visual_weight'] = 0.5
            defaults['evidence']['audio_similarity_sigma'] = 0.1
        
        # Create config from modified defaults
        sampling_config = SamplingConfig(**defaults.get('sampling', {}))
        evidence_config = EvidenceConfig(**defaults.get('evidence', {}))
        
        config = SimulationConfig(
            trial_name=defaults.get('default_trial', 'snack1'),
            sampling=sampling_config,
            evidence=evidence_config,
            **defaults.get('simulation', {})
        )
        
        # Set additional RSM-specific parameters
        config.log_dir_base = os.path.abspath(args.log_dir)
        if hasattr(args, 'max_steps_middle') and args.max_steps_middle > 0:
            config.sampling.max_steps_middle = args.max_steps_middle
            
    elif args.command == 'empirical':
        # Load defaults and create basic config for empirical analysis
        with open(defaults_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        defaults['default_trial'] = args.trial
        defaults['sampling']['cost_weight'] = 0.1
        defaults['sampling']['naive_temp'] = 0.01
        defaults['sampling']['sophisticated_temp'] = 0.01
        defaults['sampling']['max_steps'] = 25
        
        sampling_config = SamplingConfig(**defaults.get('sampling', {}))
        evidence_config = EvidenceConfig(**defaults.get('evidence', {}))
        
        config = SimulationConfig(
            trial_name=defaults.get('default_trial', 'snack1'),
            sampling=sampling_config,
            evidence=evidence_config,
            **defaults.get('simulation', {})
        )
        
        config.log_dir_base = os.path.abspath(args.log_dir)
        config.empirical_paths_file = args.paths
        config.mismatched_analysis = getattr(args, 'mismatched', False)
        
    elif args.command == 'uniform':
        # Load defaults and create config for uniform baseline
        with open(defaults_path, 'r') as f:
            defaults = yaml.safe_load(f)
        
        defaults['default_trial'] = args.trial
        defaults['sampling']['cost_weight'] = 0.1
        defaults['sampling']['naive_temp'] = 0.01
        defaults['sampling']['sophisticated_temp'] = 0.01
        defaults['sampling']['max_steps'] = args.max_steps
        
        sampling_config = SamplingConfig(**defaults.get('sampling', {}))
        evidence_config = EvidenceConfig(**defaults.get('evidence', {}))
        
        config = SimulationConfig(
            trial_name=defaults.get('default_trial', 'snack1'),
            sampling=sampling_config,
            evidence=evidence_config,
            **defaults.get('simulation', {})
        )
        
        config.log_dir_base = os.path.abspath(args.log_dir)
    
    return config


def run_rsm_simulation(args):
    """Run RSM simulation"""
    # Setup directories
    log_dir_base = os.path.abspath(args.log_dir)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Set random seed
    set_random_seed(config.seed)
    
    param_log_dir = create_param_dir(
        log_dir_base, args.trial, 
        w=args.weight, 
        naive_temp=args.naive_temp,
        soph_temp=args.soph_temp,
        max_steps=args.max_steps,
        model_type="rsm"
    )
    
    # Setup logging in the param directory
    log_path = setup_logging(param_log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting RSM simulation - Log: {log_path}")
    
    # Save metadata
    save_metadata(config, param_log_dir)
    
    # Get trials to run
    trials_to_run = get_json_files(args.trial)
    logger.info(f"Trials to run: {trials_to_run}")
    
    # Run simulation
    simulator = RSMSimulator(config, log_dir_base, param_log_dir, trials_to_run)
    results = simulator.run()

    # Generate plots automatically
    logger.info("Generating automatic plots for simulation results...")
    for trial_name in trials_to_run:
        # Remove .json extension if present
        trial_name_clean = trial_name.replace('_A1.json', '').replace('.json', '')
        try:
            create_simulation_plots(param_log_dir, trial_name_clean, args.evidence)
            logger.info(f"Generated {args.evidence} plots for trial {trial_name_clean}")
        except Exception as e:
            logger.warning(f"Could not generate plots for trial {trial_name_clean}: {e}")
    
    logger.info(f"RSM simulation completed. Results: {len(results)} trials processed")
    return results


def run_empirical_analysis(args):
    """Run empirical analysis"""
    # Setup directories  
    log_dir_base = os.path.abspath(args.log_dir)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Set random seed
    set_random_seed(config.seed)
    
    param_log_dir = create_param_dir(log_dir_base, args.trial, model_type="empirical")
    
    # Setup logging in the param directory
    log_path = setup_logging(param_log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting empirical analysis - Log: {log_path}")
    
    # Save metadata
    save_metadata(config, param_log_dir)
    
    # Get trials to run
    trials_to_run = get_json_files(args.trial)
    logger.info(f"Trials to analyze: {trials_to_run}")
    
    # Run analysis
    simulator = EmpiricalSimulator(config, log_dir_base, param_log_dir, trials_to_run)
    results = simulator.run()
    
    # Generate plots automatically (use visual evidence for empirical)
    logger.info("Generating automatic plots for empirical analysis results...")
    for trial_name in trials_to_run:
        # Remove .json extension if present
        trial_name_clean = trial_name.replace('_A1.json', '').replace('.json', '')
        try:
            create_simulation_plots(param_log_dir, trial_name_clean, 'visual')
            logger.info(f"Generated visual plots for trial {trial_name_clean}")
        except Exception as e:
            logger.warning(f"Could not generate plots for trial {trial_name_clean}: {e}")
    
    logger.info(f"Empirical analysis completed. Results: {len(results)} trials processed")
    return results


def run_uniform_baseline(args):
    """Run uniform baseline simulation"""
    # Setup directories
    log_dir_base = os.path.abspath(args.log_dir)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Set random seed
    set_random_seed(config.seed)
    
    param_log_dir = create_param_dir(log_dir_base, args.trial, max_steps=args.max_steps, model_type="uniform")
    
    # Setup logging in the param directory
    log_path = setup_logging(param_log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting uniform baseline - Log: {log_path}")
    
    # Save metadata
    save_metadata(config, param_log_dir)
    
    # Get trials to run
    trials_to_run = get_json_files(args.trial)
    logger.info(f"Trials to run: {trials_to_run}")
    
    # Run simulation
    simulator = UniformSimulator(config, log_dir_base, param_log_dir, trials_to_run)
    results = simulator.run()
    
    # Generate plots automatically (use visual evidence for uniform)
    logger.info("Generating automatic plots for uniform baseline results...")
    for trial_name in trials_to_run:
        # Remove .json extension if present
        trial_name_clean = trial_name.replace('_A1.json', '').replace('.json', '')
        try:
            create_simulation_plots(param_log_dir, trial_name_clean, 'visual')
            logger.info(f"Generated visual plots for trial {trial_name_clean}")
        except Exception as e:
            logger.warning(f"Could not generate plots for trial {trial_name_clean}: {e}")
    
    logger.info(f"Uniform baseline completed. Results: {len(results)} trials processed")
    return results


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'rsm':
            return run_rsm_simulation(args)
        elif args.command == 'empirical':
            return run_empirical_analysis(args)
        elif args.command == 'uniform':
            return run_uniform_baseline(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 