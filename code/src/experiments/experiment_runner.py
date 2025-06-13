"""
CLI interface for running experiments.

Provides command-line interface for the experiment runner.
"""

import argparse
import sys
from typing import Dict, Any

from . import ExperimentRunner


def create_cli_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Run evidence-specific RSM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s rsm --evidence audio --trial snack1
        %(prog)s rsm --evidence visual --trial snack1 --override sampling.max_steps=30
        %(prog)s batch experiments.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single RSM experiment
    rsm_parser = subparsers.add_parser('rsm', help='Run single RSM experiment')
    rsm_parser.add_argument('--evidence', 
                           choices=['visual', 'audio', 'multimodal'],
                           required=True,
                           help='Evidence type (loads corresponding YAML config)')
    rsm_parser.add_argument('--trial', 
                           required=True,
                           help='Trial name to run')
    rsm_parser.add_argument('--name',
                           help='Experiment name (optional)')
    rsm_parser.add_argument('--log-dir', 
                           default='results',
                           help='Base log directory')
    rsm_parser.add_argument('--override',
                           action='append',
                           help='Override config values (e.g., sampling.max_steps=30)')
    
    # Batch experiments from config file
    batch_parser = subparsers.add_parser('batch', help='Run batch experiments')
    batch_parser.add_argument('config_file',
                             help='YAML file defining batch experiments')
    batch_parser.add_argument('--log-dir',
                             default='results', 
                             help='Base log directory')
    
    return parser


def parse_overrides(override_args) -> Dict[str, Any]:
    """Parse CLI override arguments into nested dict"""
    overrides = {}
    
    if not override_args:
        return overrides
    
    for override in override_args:
        if '=' not in override:
            continue
            
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Try to convert value to appropriate type
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass  # Keep as string
        
        # Set nested dict value
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return overrides


def main():
    """Main CLI entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        runner = ExperimentRunner(args.log_dir)
        
        if args.command == 'rsm':
            # Parse overrides
            overrides = parse_overrides(getattr(args, 'override', None))
            
            # Run single experiment
            results = runner.run_rsm_experiment(
                evidence_type=args.evidence,
                trial_name=args.trial,
                config_overrides=overrides,
                experiment_name=getattr(args, 'name', None)
            )
            
            print(f"Experiment completed successfully. Results: {len(results)} trials")
            
        elif args.command == 'batch':
            # Load batch config and run
            import yaml
            with open(args.config_file, 'r') as f:
                batch_config = yaml.safe_load(f)
            
            results = runner.run_batch_experiments(batch_config['experiments'])
            print(f"Batch completed. {len(results)} experiments run")
            
        return 0
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 