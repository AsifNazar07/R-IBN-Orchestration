import os
import yaml
import argparse
import subprocess
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool

import networkx as nx
from munch import munchify, unmunchify


def scale_and_save_topology(base_graph, compute_scale, datarate_scale, save_path):
    """Scales compute and datarate attributes of a network graph and saves the result."""
    G = deepcopy(base_graph)

    for node in G.nodes:
        G.nodes[node]['compute'] *= compute_scale

    for u, v in G.edges:
        G[u][v]['datarate'] *= datarate_scale

    nx.write_gpickle(G, save_path)
    return str(save_path)


def generate_experiment_directory(base_exp_cfg, base_agent_cfg, out_dir, compute_scale, datarate_scale, overlay_path):
    """Creates a new experiment directory and writes updated configs and scaled overlay."""
    exp_cfg = deepcopy(base_exp_cfg)
    agent_cfg = deepcopy(base_agent_cfg)

    scenario_name = f"compute_{compute_scale}_datarate_{datarate_scale}"
    scenario_path = Path(out_dir) / scenario_name
    scenario_path.mkdir(parents=True, exist_ok=True)

    scaled_overlay_path = scenario_path / "overlay.gpickle"
    exp_cfg.overlay = str(scaled_overlay_path)

    # Write config files
    with open(scenario_path / "experiment.yml", "w") as f:
        yaml.dump(unmunchify(exp_cfg), f)
    with open(scenario_path / "agent.yml", "w") as f:
        yaml.dump(unmunchify(agent_cfg), f)

    return scenario_path, scaled_overlay_path


def spawn_job(ns):
    """Runs a single evaluation subprocess."""
    command = (
        f"python script.py --experiment {ns.experiment} "
        f"--agent {ns.agent} --episodes {ns.episodes} "
        f"--logdir {ns.logdir} --seed {ns.seed}"
    )
    subprocess.run([command], shell=True)


def main():
    parser = argparse.ArgumentParser(description="Network topology scaling scenarios for DRLGNN orchestration.")
    parser.add_argument("--compute", nargs="+", type=float, required=True, help="Scaling factors for node compute capacity")
    parser.add_argument("--datarate", nargs="+", type=float, required=True, help="Scaling factors for link datarate")
    parser.add_argument("--experiment", type=str, default="./data/experiments/abilene/trace/trace.yml")
    parser.add_argument("--agent", type=str, default="./data/configurations/random.yml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="./results/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=1)

    args = parser.parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(args.experiment, "r") as f:
        base_exp_cfg = munchify(yaml.safe_load(f))
    with open(args.agent, "r") as f:
        base_agent_cfg = munchify(yaml.safe_load(f))

    original_overlay = nx.read_gpickle(base_exp_cfg.overlay)

    jobs = []
    for comp_scale, data_scale in zip(args.compute, args.datarate):
        scenario_dir, new_overlay_path = generate_experiment_directory(
            base_exp_cfg, base_agent_cfg, logdir, comp_scale, data_scale, base_exp_cfg.overlay
        )
        scale_and_save_topology(original_overlay, comp_scale, data_scale, new_overlay_path)

        run_ns = deepcopy(args)
        run_ns.logdir = str(scenario_dir)
        run_ns.experiment = str(scenario_dir / "experiment.yml")
        run_ns.agent = str(scenario_dir / "agent.yml")

        with open(scenario_dir / "args.yml", "w") as f:
            yaml.dump(vars(run_ns), f)

        jobs.append(run_ns)

    with Pool(processes=args.pool) as pool:
        pool.map(spawn_job, jobs)


if __name__ == "__main__":
    main()
