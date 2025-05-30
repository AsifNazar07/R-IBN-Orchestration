import numpy as np
from typing import Dict, Tuple

class ResourceManager:
    """
    A class that encapsulates the logic for computing resource usage (CPU, memory)
    based on VNF configurations and current traffic loads.
    """

    def __init__(self, vnfs: list, max_compute: float, max_memory: float):
        self.vnfs = vnfs
        self.max_compute = max_compute
        self.max_memory = max_memory

    def compute_resource_demand(self, vnf_type: int, rate: float) -> Tuple[float, float]:
        """
        Compute the compute and memory demand for a given VNF and input rate.

        Parameters:
            vnf_type (int): Index of the VNF in the configuration list.
            rate (float): Input datarate (in MB/s).

        Returns:
            Tuple[float, float]: Required (CPU, Memory) resources.
        """
        config = self.vnfs[vnf_type]

        if rate <= 0:
            return 0.0, 0.0

        if rate > config['max. req_transf_rate']:
            return np.inf, np.inf

        normalized_rate = rate / config['scale']

        compute = config['coff'] + config['ccoef_1'] * normalized_rate + \
                  config['ccoef_2'] * normalized_rate**2 + \
                  config['ccoef_3'] * normalized_rate**3 + \
                  config['ccoef_4'] * normalized_rate**4

        memory = config['moff'] + config['mcoef_1'] * normalized_rate + \
                 config['mcoef_2'] * normalized_rate**2 + \
                 config['mcoef_3'] * normalized_rate**3 + \
                 config['mcoef_3'] * normalized_rate**4

        return max(0.0, compute), max(0.0, memory)

    def validate_placement(self, node: int, vnf_type: int, rate: float,
                           node_resources: Dict[int, Dict[str, float]]) -> bool:
        """
        Validates if a VNF of given type can be placed on a node with current residuals.

        Parameters:
            node (int): Node index.
            vnf_type (int): Index of the VNF.
            rate (float): Required traffic rate for the VNF.
            node_resources (Dict[int, Dict]): Dict of available resources per node.

        Returns:
            bool: Whether placement is valid.
        """
        compute, memory = self.compute_resource_demand(vnf_type, rate)
        available = node_resources.get(node, {})

        return compute <= available.get('compute', 0) and memory <= available.get('memory', 0)

    def update_resources(self, node: int, vnf_type: int, rate: float,
                         node_resources: Dict[int, Dict[str, float]]) -> None:
        """
        Updates resource dictionary after VNF deployment.

        Parameters:
            node (int): Node index.
            vnf_type (int): Type of VNF being deployed.
            rate (float): Required traffic rate.
            node_resources (Dict[int, Dict]): Resource dict to be modified in-place.
        """
        compute, memory = self.compute_resource_demand(vnf_type, rate)
        node_resources[node]['compute'] -= compute
        node_resources[node]['memory'] -= memory
