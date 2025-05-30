import numpy as np
import torch as th
from copy import deepcopy
from itertools import combinations_with_replacement
from collections import Counter
import numpy.ma as ma
import networkx as nx

from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from coordination.environment.traffic import Request


class RandomPolicy:
    def __init__(self, seed=None, **kwargs):
        self.rng = np.random.default_rng(seed)

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        valid_nodes = list(env.valid_routes.keys())
        return self.rng.choice(valid_nodes) if valid_nodes else env.REJECT_ACTION


class AllCombinations:
    def __init__(self, **kwargs):
        self.actions = []

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        if not self.actions:
            self.actions = self._generate_combinations(env)
        return self.actions.pop(0) if self.actions else env.REJECT_ACTION

    def _generate_combinations(self, env):
        nodes = list(env.net.nodes())
        vtypes = env.request.vtypes
        best_score, best_comb = float('inf'), None

        compute_avail = sum(env.computing.values())
        memory_avail = sum(env.memory.values())
        link_avail = sum(env.datarate.values())

        for placement in combinations_with_replacement(nodes, len(vtypes)):
            sim_env = deepcopy(env)
            for node in placement:
                if node not in sim_env.valid_routes:
                    break
                _, _, done, _ = sim_env.step(node)
                if done:
                    break

            if not sim_env.done:
                delta_c = compute_avail - sum(sim_env.computing.values())
                delta_m = memory_avail - sum(sim_env.memory.values())
                delta_d = link_avail - sum(sim_env.datarate.values())
                score = delta_c + delta_m + delta_d

                if score < best_score:
                    best_score, best_comb = score, placement

        return list(best_comb) if best_comb else [env.REJECT_ACTION]


class GreedyHeuristic:
    def __init__(self, **kwargs):
        pass

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        valid_nodes = list(env.valid_routes.keys())
        if not valid_nodes:
            return env.REJECT_ACTION

        src = env.routes_bidict[env.request][-1][1]  # get current node
        try:
            lengths, _ = nx.single_source_dijkstra(
                env.net, source=src, weight=env.get_weights, cutoff=env.request.resd_lat
            )
            return min(valid_nodes, key=lambda n: lengths.get(n, float('inf')))
        except nx.NetworkXNoPath:
            return env.REJECT_ACTION


class MaskedPPO(PPO):
    def predict(self, observation, deterministic=False, env=None, **kwargs):
        obs_tensor = th.as_tensor(observation).float().unsqueeze(0)
        if deterministic and env is not None:
            valid_mask = np.zeros(env.ACTION_DIM, dtype=bool)
            valid_mask[list(env.valid_routes.keys())] = True

            latent_pi, _, latent_sde = self.policy._get_latent(obs_tensor)
            dist = self.policy._get_action_dist_from_latent(latent_pi, latent_sde)

            logits = dist.distribution.logits.detach().cpu().numpy().squeeze()
            masked_logits = ma.masked_array(logits, mask=~valid_mask, fill_value=np.NINF)
            return masked_logits.argmax()

        action, _ = super().predict(obs_tensor, deterministic=deterministic)
        return action.item()

    def load(self, path, device='auto'):
        self.policy = MlpPolicy.load(path, device=device)
        self.learn = lambda *args, **kwargs: None
