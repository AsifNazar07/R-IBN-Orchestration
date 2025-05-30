import numpy as np
import pandas as pd
import os
from timeit import default_timer as timer
from coordination.environment.traffic import Traffic


def evaluate_episode(agent, monitor, process):
    """Runs one full episode and returns statistics."""
    start = timer()
    ep_reward = 0.0
    obs = monitor.reset()

    while not monitor.env.done:
        action = agent.predict(observation=obs, env=monitor.env, process=process, deterministic=True)
        obs, reward, _, _ = monitor.step(action)
        ep_reward += reward

    end = timer()
    ep_results = monitor.get_episode_results()
    ep_results['time'] = end - start

    return ep_results


def save_ep_results(data: dict, path):
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index().rename(columns={'index': 'episode'})
    path = path / 'results.csv'

    if not os.path.isfile(str(path)):
        df.to_csv(path, header=True, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)
