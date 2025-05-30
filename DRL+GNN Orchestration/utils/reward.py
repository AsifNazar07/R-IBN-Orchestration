import numpy as np

class RewardCalculator:
    """
    Computes reward signals for DRL-based orchestration agents.
    Modularized for DRL+GNN pipelines.
    """

    def __init__(self, mode='sparse', weights=None):
        """
        Initialize the reward calculator.

        Args:
            mode (str): Type of reward signal ('sparse', 'dense', 'custom').
            weights (dict): Optional dictionary to weight reward components.
        """
        self.mode = mode
        self.weights = weights or {
            'accept': 1.0,
            'latency': 0.2,
            'resource_efficiency': 0.3,
            'balance': 0.2,
            'penalty': -1.0
        }

    def compute(self, finalized: bool, deployed: bool, request, env) -> float:
        """
        Compute the reward based on the current request and environment state.

        Args:
            finalized (bool): Whether the request lifecycle is finalized.
            deployed (bool): Whether the request was successfully deployed.
            request (Request): The request object.
            env (ServiceCoordination): The environment instance.

        Returns:
            float: Calculated reward value.
        """
        if self.mode == 'sparse':
            return self._sparse_reward(finalized, deployed)
        elif self.mode == 'dense':
            return self._dense_reward(finalized, deployed, request, env)
        elif self.mode == 'custom':
            return self._custom_reward(finalized, deployed, request, env)
        else:
            raise ValueError(f"Unsupported reward mode: {self.mode}")

    def _sparse_reward(self, finalized, deployed):
        return 1.0 if deployed else 0.0

    def _dense_reward(self, finalized, deployed, request, env):
        if not finalized:
            return 0.0

        reward = 0.0
        if deployed:
            reward += self.weights['accept']

            # Latency efficiency
            if request.max_latency > 0:
                reward += self.weights['latency'] * (1.0 - request.resd_lat / request.max_latency)

            # Resource efficiency (lower is better)
            r = env.occupied
            total = r['compute'] + r['memory'] + r['datarate']
            if total > 0:
                efficiency = 1.0 / total
                reward += self.weights['resource_efficiency'] * efficiency

            # Balance (optional: reward lower variance across node utilizations)
            cpu_util = [v for v in env.computing.values()]
            mem_util = [v for v in env.memory.values()]
            var_cpu = np.var(cpu_util)
            var_mem = np.var(mem_util)
            balance = 1.0 / (1.0 + var_cpu + var_mem)
            reward += self.weights['balance'] * balance
        else:
            reward += self.weights['penalty']

        return reward

    def _custom_reward(self, finalized, deployed, request, env):
        # Placeholder for user-defined reward strategies
        raise NotImplementedError("Custom reward mode not implemented.")
