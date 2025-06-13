"""
Command Line Interface for simulation.
Provides a CLI that delegates all orchestration to SimulationRunner.
"""

import argparse
import sys
from typing import Dict, Any
from ..sim import SimulationRunner


def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s rsm --evidence visual --trial snack1 --override sampling.max_steps=25 sampling.cost_weight=10.0
        %(prog)s empirical --paths results/paths.csv --trial snack1 --mismatched
        %(prog)s uniform --evidence audio --trial snack2 --max-steps 30
        %(prog)s batch experiments.yaml
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Simulation type')
    
    # RSM simulation
    rsm_parser = subparsers.add_parser('rsm', help='Run RSM simulation')
    rsm_parser.add_argument('--evidence', choices=['visual', 'audio', 'multimodal'], 
                           required=True, help='Evidence type')
    rsm_parser.add_argument('--trial', required=True, help='Trial name (or "all")')
    rsm_parser.add_argument('--log-dir', default='results', help='Log directory')
    rsm_parser.add_argument('--name', help='Experiment name (optional)')
    rsm_parser.add_argument('--override', action='append', 
                           help='Override config parameter using dot notation (e.g., sampling.naive_temp=0.1)')
    
    # Empirical analysis  
    emp_parser = subparsers.add_parser('empirical', help='Run empirical analysis')
    emp_parser.add_argument('--paths', required=True, help='CSV file with empirical paths')
    emp_parser.add_argument('--trial', default='all', help='Trial name (or "all")')
    emp_parser.add_argument('--mismatched', action='store_true', help='Mismatched analysis')
    emp_parser.add_argument('--log-dir', default='results', help='Log directory')
    emp_parser.add_argument('--name', help='Experiment name (optional)')
    emp_parser.add_argument('--override', action='append', 
                           help='Override config parameter using dot notation')
    
    # Uniform baseline
    uniform_parser = subparsers.add_parser('uniform', help='Run uniform baseline')
    uniform_parser.add_argument('--evidence', choices=['visual', 'audio', 'multimodal'], 
                               required=True, help='Evidence type')
    uniform_parser.add_argument('--trial', required=True, help='Trial name (or "all")')
    uniform_parser.add_argument('--max-steps', type=int, default=25, help='Maximum steps')
    uniform_parser.add_argument('--log-dir', default='results', help='Log directory')
    uniform_parser.add_argument('--name', help='Experiment name (optional)')
    uniform_parser.add_argument('--override', action='append', 
                               help='Override config parameter using dot notation')
    
    # Batch experiments
    batch_parser = subparsers.add_parser('batch', help='Run batch experiments')
    batch_parser.add_argument('config_file', help='YAML file defining batch experiments')
    batch_parser.add_argument('--log-dir', default='results', help='Log directory')
    
    return parser


def parse_overrides(override_args) -> Dict[str, Any]:
    """Parse CLI override arguments into parameter dictionary"""
    overrides = {}
    
    if not override_args:
        return overrides
    
    for override in override_args:
        if '=' not in override:
            continue
            
        key, value = override.split('=', 1)

        try:
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass  # Keep as string
        
        overrides[key] = value
    
    return overrides


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        runner = SimulationRunner(args.log_dir)
        
        if args.command == 'rsm':
            # Parse overrides
            overrides = parse_overrides(getattr(args, 'override', None))
            
            # Run RSM experiment
            results = runner.run_rsm_experiment(
                evidence_type=args.evidence,
                trial_name=args.trial,
                config_overrides=overrides,
                experiment_name=getattr(args, 'name', None)
            )
            
            print(f"RSM simulation completed. Results: {len(results)} trials processed")
            
        elif args.command == 'empirical':
            # Parse overrides
            overrides = parse_overrides(getattr(args, 'override', None))
            
            # Run empirical analysis
            results = runner.run_empirical_experiment(
                empirical_paths_file=args.paths,
                trial_name=args.trial,
                mismatched_analysis=args.mismatched,
                config_overrides=overrides,
                experiment_name=getattr(args, 'name', None)
            )
            
            print(f"Empirical analysis completed. Results: {len(results)} trials processed")
            
        elif args.command == 'uniform':
            # Parse overrides
            overrides = parse_overrides(getattr(args, 'override', None))
            
            # Run uniform baseline
            results = runner.run_uniform_experiment(
                evidence_type=args.evidence,
                trial_name=args.trial,
                max_steps=args.max_steps,
                config_overrides=overrides,
                experiment_name=getattr(args, 'name', None)
            )
            
            print(f"Uniform baseline completed. Results: {len(results)} trials processed")
            
        elif args.command == 'batch':
            # Load batch config and run
            import yaml
            with open(args.config_file, 'r') as f:
                batch_config = yaml.safe_load(f)
            
            results = runner.run_batch_experiments(batch_config.get('experiments', []))
            print(f"Batch completed. {len(results)} experiments run")
            
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 