import os
import yaml
import argparse
import tempfile
import subprocess
from pathlib import Path
from copy import deepcopy
from itertools import product
from multiprocessing import Pool

from munch import munchify, unmunchify


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for planning-based DRLGNN agents.")
    parser.add_argument('--experiment', type=str, default='./data/experiments/abilene/trace/trace.yml')
    parser.add_argument('--agent', type=str, default='./data/configurations/futurecoord.yml')
    parser.add_argument('--oracle', action='store_true', help="Enable oracle traffic forecast during simulation")
    parser.add_argument('--logdir', type=str, default='./results/search_grid')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=1)

    parser.add_argument('--max_searches', nargs='+', type=int, required=True, help="Grid values for max search iterations")
    parser.add_argument('--max_requests', nargs='+', type=int, required=True, help="Grid values for max forecasted flows")
    parser.add_argument('--sim_discounts', nargs='+', type=float, required=True, help="Grid values for simulation rollout discounting")

    return parser.parse_args()


def launch_simulation(config, args, logdir):
    """Launches a single simulation configuration using a temporary YAML config."""
    policy = config.policy
    run_id = f"search_{policy.max_searches}_req_{policy.max_requests}_disc_{policy.sim_discount:.2f}_oracle_{args.oracle}"
    run_dir = logdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write updated config to temp file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".yml") as tmp_file:
        yaml.dump(unmunchify(config), tmp_file)
        tmp_path = tmp_file.name

    # Construct command
    cmd = (
        f"python script.py "
        f"--experiment {args.experiment} "
        f"--agent {tmp_path} "
        f"--episodes {args.episodes} "
        f"--logdir {run_dir} "
        f"--seed {args.seed}"
    )
    if args.oracle:
        cmd += " --oracle"

    subprocess.run(cmd, shell=True)
    os.unlink(tmp_path)


def main():
    args = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Load base agent config
    with open(args.agent, 'r') as f:
        base_config = munchify(yaml.safe_load(f))

    # Create all combinations of hyperparameters
    job_configs = []
    for ms, mr, disc in product(args.max_searches, args.max_requests, args.sim_discounts):
        config = deepcopy(base_config)
        config.policy.max_searches = ms
        config.policy.max_requests = mr
        config.policy.sim_discount = disc
        job_configs.append((config, args, logdir))

    # Run in parallel
    with Pool(processes=args.pool) as pool:
        pool.starmap(launch_simulation, job_configs)


if __name__ == "__main__":
    main()
