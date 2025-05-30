import argparse
import yaml
import numpy as np
from pathlib import Path
from munch import munchify
from stable_baselines3.common.monitor import Monitor

from coordination.environment.deployment import ServiceCoordination
from coordination.environment.traffic import Traffic
from coordination.evaluation.monitor import CoordMonitor
from coordination.evaluation.utils import evaluate_episode, setup_agent, setup_process


def parse_args():
    parser = argparse.ArgumentParser(description="Main execution script for DRLGNN coordination experiments.")
    parser.add_argument('--experiment', type=str, required=True, help='Path to experiment configuration YAML')
    parser.add_argument('--agent', type=str, required=True, help='Path to agent configuration YAML')
    parser.add_argument('--logdir', type=str, required=True, help='Directory to store logs and results')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--oracle', action='store_true', help='Enable oracle traffic forecast')

    return parser.parse_args()


def main():
    args = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Load configuration files
    with open(args.experiment, 'r') as f:
        exp_cfg = munchify(yaml.safe_load(f))
    with open(args.agent, 'r') as f:
        agent_cfg = munchify(yaml.safe_load(f))

    rng = np.random.default_rng(args.seed)

    # Prepare traffic generation process
    traffic = setup_process(
        rng=rng,
        exp=exp_cfg,
        services=exp_cfg.services,
        eday=0,
        sdays=list(range(len(exp_cfg.services))),
        load=exp_cfg.sim_load,
        rate=exp_cfg.sim_datarate,
        latency=exp_cfg.sim_latency
    )

    # Build simulation environment
    env = ServiceCoordination(
        net_path=exp_cfg.overlay,
        process=traffic,
        vnfs=exp_cfg.vnfs,
        services=exp_cfg.services
    )

    # Wrap environment with monitor
    monitor = CoordMonitor(
        episode=0,
        tag="DRLGNN",
        env=env,
        filename=str(logdir)
    )

    # Setup agent
    agent = setup_agent(agent_cfg, env, seed=args.seed)

    # Evaluate over N episodes
    all_results = []
    for ep in range(args.episodes):
        monitor.episode = ep
        results = evaluate_episode(agent, monitor, traffic)
        all_results.append(results)

    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(logdir / "results.csv", index=False)
    print(f"[✔] Evaluation complete. Results saved to: {logdir}/results.csv")


if __name__ == "__main__":
    main()
