import argparse
import logging
import os
import random
import numpy as np
import json
from dataclasses import asdict

from utils import create_param_dir, get_json_files 
from simulation import RSMSimulator, EmpiricalSimulator, UniformSimulator
from params import SimulationParams


def parse_args():
    parser = argparse.ArgumentParser("Simulation argument parser")
    subparser = parser.add_subparsers(dest="command", required=True)

    # RSM
    simulate = subparser.add_parser("rsm", help="Run recursive simulation model")
    simulate.add_argument('--trial', type=str, default='snack1', help='which trial to run (default: `snack1`)')
    simulate.add_argument('--w', type=float, default=0.1, help='weight parameter (default: 0.1)')
    simulate.add_argument('--n-temp', type=float, default=0.01, help='softmax temperature parameter for naive agent (default: 0.01)')
    simulate.add_argument('--s-temp', type=float, default=0.01, help='softmax temperature parameter for sophisticated agent (default: 0.01)')
    simulate.add_argument('--max-steps', type=int, default=25, help='maximum number of steps for subgoal simple paths (default: 25)')
    simulate.add_argument('--log-dir', type=str, default='../../results', help='data logging directory for model (default: `../../results`)')

    # Empirical
    empirical = subparser.add_parser("empirical", help="Run empirical model")
    empirical.add_argument('--trial', type=str, default='snack1', help='which trial to run (default: `snack1`)')
    empirical.add_argument('--paths', type=str, default='../../data/exp2_suspect/humans/human_paths.csv', help='Path to CSV containing empirical paths (default: `../../data/exp2_suspect/humans/human_paths.csv`)')
    empirical.add_argument('--log-dir', type=str, default='../../results', help='data logging directory for model (default: `../../results`)')
    empirical.add_argument('--mismatched', action='store_true', help='Calculate naive preds using soph paths, and vice versa (default: False)')

    # Uniform
    uniform = subparser.add_parser("uniform", help="Run uniform model")
    uniform.add_argument('--trial', type=str, default='snack1', help='which trial to run (default: `snack1`)')
    uniform.add_argument('--max-steps', type=int, default=25, help='maximum number of steps for subgoal simple paths (default: 25)')
    uniform.add_argument('--log-dir', type=str, default='../../results', help='data logging directory for model (default: `../../results`)')

    return parser.parse_args()


def logger(args):
    logger = logging.getLogger(__name__)
    log_filename = ""
    
    log_dir_base = args.log_dir
    if not os.path.exists(log_dir_base):
        os.makedirs(log_dir_base)
    
    trial_id_for_log = args.trial if args.trial != 'all' else "all_trials"

    if args.command == 'rsm':
        param_log_dir = create_param_dir(log_dir_base, trial_id_for_log, args.w, args.n_temp, args.s_temp, args.max_steps,model_type="rsm")
        log_filename = os.path.join(param_log_dir, 'simulation.log')

    elif args.command == 'empirical':
        param_log_dir = create_param_dir(log_dir_base, trial_id_for_log, model_type="empirical")
        log_filename = os.path.join(param_log_dir, 'empirical.log')
        if not os.path.exists(args.paths):
            logger.error(f"Error: Empirical path file not found: {args.paths}")
            exit(1)

    elif args.command == 'uniform':
        param_log_dir = create_param_dir(
            log_dir_base, trial_id_for_log,
            max_steps=args.max_steps,
            model_type="uniform"
        )
        log_filename = os.path.join(param_log_dir, 'uniform_simulation.log')

    else:
        logger.error(f"Unknown command: {args.command}")
        exit(1)

    metadata_filepath = os.path.join(param_log_dir, 'metadata.json')
    metadata_content = {
        "trial": args.trial,
        "command": args.command,
        "parameters": vars(args)
    }
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata_content, f, indent=4)
    logger.info(f"Metadata saved to {metadata_filepath}")

    if os.path.exists(log_filename): open(log_filename, 'w').close()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logger.info(f"Starting {args.command} process...")
    logger.info(f"Results directory: {param_log_dir}")
    
    try:
        trials_to_run = get_json_files(args.trial)
        logger.info(f"Trials: {', '.join(t.split('_A1.json')[0] for t in trials_to_run)}")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Fatal: {e}")
        exit(1)
    
    return log_dir_base, param_log_dir, trials_to_run


if __name__ == '__main__':
    args = parse_args()

    log_dir_base, param_log_dir, trials_to_run = logger(args)
    
    main_logger = logging.getLogger(__name__)
    experiment_params = vars(args)

    params = SimulationParams(**experiment_params)
    main_logger.info(f"Processed Simulation parameters: {params}")

    metadata_filepath = os.path.join(param_log_dir, 'metadata.json')
    params_dict_for_metadata = asdict(params)
    params_dict_for_metadata.pop('naive_A_crumb_likelihoods_map', None)
    params_dict_for_metadata.pop('naive_B_crumb_likelihoods_map', None)
    params_dict_for_metadata.pop('paths', None)

    with open(metadata_filepath, 'w') as f_meta:
        json.dump(params_dict_for_metadata, f_meta, indent=4)

    np.random.seed(params.seed)
    random.seed(params.seed)

    if args.command == 'rsm':
        simulator = RSMSimulator(args, log_dir_base, param_log_dir, params, trials_to_run)
    elif args.command == 'empirical':
        simulator = EmpiricalSimulator(args, log_dir_base, param_log_dir, params, trials_to_run)
    elif args.command == 'uniform':
        simulator = UniformSimulator(args, log_dir_base, param_log_dir, params, trials_to_run)
    else:
        main_logger.error(f"Unknown command: {args.command}. Exiting.")
        exit(1)
    
    # simulator.run(params)
    simulator.run()
    main_logger.info(f"Simulation completed for {args.command} model.")
