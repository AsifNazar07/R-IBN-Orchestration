import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from coordination.environment.deployment import ServiceCoordination
from utils.state_encoder import GraphStateEncoder
from utils.reward import RewardStrategy
from utils.routing import RoutingStrategy
from utils.resources import ResourceManager


class GNNPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(GNNPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DRLGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(DRLGNNExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.encoder = GraphStateEncoder(input_dim=input_dim, hidden_dim=features_dim)

    def forward(self, observations):
        return self.encoder(observations)


def setup_drlgnn_agent(config, env, seed=0):
    policy_kwargs = dict(
        features_extractor_class=DRLGNNExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.policy.learning_rate,
        n_steps=config.policy.n_steps,
        batch_size=config.policy.batch_size,
        n_epochs=config.policy.n_epochs,
        gamma=config.policy.gamma,
        gae_lambda=config.policy.gae_lambda,
        clip_range=config.policy.clip_range,
        ent_coef=config.policy.ent_coef,
        vf_coef=config.policy.vf_coef,
        max_grad_norm=config.policy.max_grad_norm,
        tensorboard_log=config.policy.tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1
    )
    return model


class DRLGNNOrchestrator:
    def __init__(self, config, env_config):
        self.config = config
        self.env_config = env_config

        self.reward_module = RewardStrategy(**config.reward)
        self.routing_module = RoutingStrategy(**config.routing)
        self.resource_module = ResourceManager(**config.resource)

        self.env = DummyVecEnv([
            lambda: ServiceCoordination(
                net_path=env_config.topology,
                process=env_config.traffic,
                vnfs=env_config.vnfs,
                services=env_config.services
            )
        ])

        self.agent = setup_drlgnn_agent(config, self.env, seed=config.seed)

    def train(self):
        self.agent.learn(total_timesteps=self.config.train.timesteps)

    def evaluate(self, env_eval):
        obs = env_eval.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = env_eval.step(action)
            total_reward += reward
        return total_reward

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent = PPO.load(path, env=self.env)
