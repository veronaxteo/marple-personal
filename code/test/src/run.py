import argparse
from pathlib import Path

from src.configs import SimulationConfig
from src.core.simulation import RSMSimulator
from src.utils.io_utils import create_param_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",   default=Path("src/configs/default.yaml"))
    p.add_argument("--extra", nargs="*", default=[], metavar="KEY=VAL",
                   help="CLI overrides, e.g. sampling.num_detective_paths=5000")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    cfg = SimulationConfig.from_cli(cfg_file=args.cfg, overrides=args.extra)

    log_dir_base = cfg.log_dir
    param_log_dir = create_param_dir(log_dir_base, cfg.trial_name, cfg)

    sim = RSMSimulator(log_dir_base=log_dir_base, param_log_dir=param_log_dir, args=args, cfg=cfg, trials_to_run=[cfg.trial_name])
    sim.run()

if __name__ == "__main__":
    main()
