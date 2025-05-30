import os
import yaml
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from munch import munchify, unmunchify

def spawn(ns):
    command = f"python script.py --experiment {ns.experiment} --agent {ns.agent} --episodes {ns.episodes} --logdir {ns.logdir} --seed {ns.seed}"
    if ns.oracle:
        command += " --oracle"
    if ns.save_model:
        command += f" --save_model {ns.save_model}"

    subprocess.run([command], shell=True)


parser = argparse.ArgumentParser(description="Run DRLGNN experiment sensitivity analysis")
parser.add_argument('--factors', nargs='+', type=float, required=True, help="List of primary scaling factors")
parser.add_argument('--sim_factors', nargs='+', type=float, required=True, help="List of simulated scaling factors")
parser.add_argument('--property', type=str, choices=['load', 'datarate', 'latency'], default='load')
parser.add_argument('--traffic', type=str, choices=['accurate', 'erroneous'], default='accurate')
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pool', type=int, default=1)
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--experiment', type=str, default='./data/experiments/abilene/trace.yml')
parser.add_argument('--agent', type=str, default='./data/configurations/drlgnn.yml')
parser.add_argument('--logdir', type=str, default='./results/DRLGNN/')
parser.add_argument('--save_model', type=str, default='')

if __name__ == '__main__':
    args = parser.parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(args.experiment, 'r') as file:
        experiment = munchify(yaml.safe_load(file))
    with open(args.agent, 'r') as file:
        agent = munchify(yaml.safe_load(file))

    namespace = deepcopy(args)
    del namespace.factors
    del namespace.sim_factors
    del namespace.property
    del namespace.pool

    for factor, sim_factor in zip(args.factors, args.sim_factors):
        exp = deepcopy(experiment)
        path = logdir / f"factor_{factor}_sim_{sim_factor}"
        path.mkdir(parents=True, exist_ok=True)

        exp.traffic = args.traffic
        if args.property == 'load':
            exp.load = factor
            exp.sim_load = sim_factor
        elif args.property == 'datarate':
            exp.datarate = factor
            exp.sim_datarate = sim_factor
        elif args.property == 'latency':
            exp.latency = factor
            exp.sim_latency = sim_factor
        else:
            raise ValueError("Unsupported property")

        with open(path / 'experiment.yml', 'w') as file:
            yaml.dump(unmunchify(exp), file)

        with open(path / 'agent.yml', 'w') as file:
            yaml.dump(unmunchify(agent), file)

        namespace.logdir = str(path)
        namespace.experiment = str(path / 'experiment.yml')
        namespace.agent = str(path / 'agent.yml')
        namespace.save_model = str(path / 'model.zip') if args.save_model else ''

        with open(path / 'args.yml', 'w') as file:
            yaml.dump(vars(namespace), file)

    runs = []
    for run in os.listdir(str(logdir)):
        with open(logdir / run / 'args.yml', 'r') as file:
            runs.append(munchify(yaml.safe_load(file)))

    with Pool(processes=args.pool) as pool:
        pool.map(spawn, runs)
