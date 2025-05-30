import numpy as np
from typing import Dict, Any

class StateEncoder:
    """
    Encodes the environment state into a fixed-size vector.
    This modular encoder handles graph, node, and request-level information.
    """

    def __init__(self, graph, vnfs: list, services: list):
        self.graph = graph
        self.vnfs = vnfs
        self.services = services
        self.num_services = len(services)
        self.num_vnfs = len(vnfs)
        self.num_nodes = graph.number_of_nodes()
        self.node_feature_size = 7

    def encode(self, env_state: Dict[str, Any]) -> np.ndarray:
        node_features = [
            self._encode_node(node_id, env_state) for node_id in self.graph.nodes
        ]
        node_features_flat = np.concatenate(node_features)

        service_features = self._encode_service(env_state)
        graph_stats = self._encode_global_stats(env_state)

        return np.concatenate([node_features_flat, service_features, graph_stats])

    def _encode_node(self, node_id: int, env_state: Dict[str, Any]) -> np.ndarray:
        valid = float(node_id in env_state['valid_routes'])
        latency = env_state['latency'].get(node_id, 1.0)
        hops = env_state['hops'].get(node_id, 1.0)
        cdemand = env_state['cdemand'].get(node_id, 1.0)
        mdemand = env_state['mdemand'].get(node_id, 1.0)
        resd_comp = env_state['resd_comp'].get(node_id, 0.0)
        resd_mem = env_state['resd_mem'].get(node_id, 0.0)
        return np.array([valid, latency, hops, cdemand, mdemand, resd_comp, resd_mem])

    def _encode_service(self, env_state: Dict[str, Any]) -> np.ndarray:
        request = env_state['request']
        vtypes = request.get('vtypes', [])
        vnum = len(vtypes)

        stype = np.zeros(self.num_services)
        stype[request['service']] = 1.0

        counter = {v: vtypes.count(v) / len(vtypes) for v in set(vtypes)}
        vnf_counts = np.zeros(self.num_vnfs)
        for v, count in counter.items():
            vnf_counts[v] = count

        datarate = request['datarate'] / env_state['max_linkrate']
        resd_lat = request['resd_lat'] / env_state['max_latency']

        egress_enc = np.zeros(self.num_nodes)
        egress_enc[request['egress']] = 1.0

        return np.concatenate([
            np.array([env_state['crelease'], env_state['mrelease']]),
            stype, vnf_counts, np.array([datarate, resd_lat]), egress_enc
        ])

    def _encode_global_stats(self, env_state: Dict[str, Any]) -> np.ndarray:
        num_deployed = np.array(env_state['vnf_deployments']) / self.num_nodes
        return np.concatenate([
            num_deployed,
            np.array([
                env_state['mean_cutil'],
                env_state['mean_mutil'],
                env_state['mean_dutil']
            ])
        ])
