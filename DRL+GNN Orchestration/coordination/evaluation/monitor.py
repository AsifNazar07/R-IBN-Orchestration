import numpy as np
from munch import unmunchify
from tensorboardX import SummaryWriter
from stable_baselines3.common.monitor import Monitor


class CoordMonitor(Monitor):
    """Extended Monitor for DRL+GNN-based coordination environments."""
    
    REQUEST_KEYS = [
        'accepts', 'requests', 'num_invalid', 'num_rejects',
        'no_egress_route', 'no_extension', 'skipped_on_arrival'
    ]
    ACCEPTED_KEYS = [
        'cum_service_length', 'cum_route_hops', 'cum_datarate',
        'cum_max_latency', 'cum_resd_latency'
    ]
    ACCEPTED_VALS = [
        'mean_service_len', 'mean_hops', 'mean_datarate',
        'mean_latency', 'mean_resd_latency'
    ]

    def __init__(self, episode: int, tag: str, env, filename: str = None,
                 allow_early_resets=True, reset_keywords=(), info_keywords=()):
        super().__init__(env, None, allow_early_resets, reset_keywords, info_keywords)
        self.writer = SummaryWriter(filename)
        self.episode = episode
        self.tag = tag

        self.c_util = []
        self.m_util = []
        self.d_util = []

    def close(self):
        self.writer.flush()
        self.writer.close()
        super().close()

    def reset(self, **kwargs):
        self.c_util.clear()
        self.m_util.clear()
        self.d_util.clear()
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._log_per_service_stats()
        self._update_utilization()
        return obs, reward, done, info

    def _log_per_service_stats(self):
        for service in range(len(self.env.services)):
            logs = unmunchify(self.env.info[service])
            for key in self.REQUEST_KEYS:
                value = logs[key] / max(self.env.num_requests, 1)
                self.writer.add_scalar(f'{self.tag}/{service}/{key}', value, self.episode)

            accepts = logs['accepts'] or np.inf
            for key in self.ACCEPTED_KEYS:
                value = logs[key] / accepts
                self.writer.add_scalar(f'{self.tag}/{service}/{key}', value, self.episode)

    def _update_utilization(self):
        nodes = self.env.net.nodes
        edges = self.env.net.edges

        cutil = [1 - self.env.computing[n] / nodes[n]['compute'] for n in nodes]
        mutil = [1 - self.env.memory[n] / nodes[n]['memory'] for n in nodes]
        self.c_util.append(np.mean(cutil))
        self.m_util.append(np.mean(mutil))

        max_cap = [self.env.net.edges[e]['datarate'] for e in edges]
        cap = [self.env.datarate[frozenset(e)] for e in edges]
        dutil = 1 - np.asarray(cap) / np.asarray(max_cap)
        self.d_util.append(np.mean(dutil))

    def get_episode_results(self):
        ep_stats = {}
        info = [unmunchify(i) for i in self.env.info]
        total_accepts = max(1, sum(s['accepts'] for s in info))
        total_requests = max(1, self.env.num_requests)

        ep_stats['accept_rate'] = total_accepts / total_requests
        ep_stats['balanced_accept_rate'] = np.prod([s['accepts'] / max(s['requests'], 1) for s in info])

        for key, val in zip(self.ACCEPTED_KEYS, self.ACCEPTED_VALS):
            agg = sum(s[key] for s in info)
            ep_stats[val] = agg / total_accepts

        for service, logs in enumerate(info):
            for key in self.REQUEST_KEYS:
                ep_stats[f'serivce_{service}_{key}'] = logs[key]
            for key, val in zip(self.ACCEPTED_KEYS, self.ACCEPTED_VALS):
                ep_stats[f'serivce_{service}_{val}'] = logs[key]

        ep_stats.update({
            'mean_cutil': np.mean(self.c_util),
            'mean_mutil': np.mean(self.m_util),
            'mean_dutil': np.mean(self.d_util),
            'ep_return': self.get_episode_rewards()[0],
            'ep_length': self.get_episode_lengths()[0],
            'ep_time': self.get_episode_times()[0],
        })

        return ep_stats
