import networkx as nx
import numpy as np

class RoutingUtils:
    """
    Encapsulates routing computations such as shortest-path filtering,
    latency-bound path extraction, and link utilization checks.
    """

    def __init__(self, graph, propagation_delays, datarate_residuals):
        """
        Parameters:
            graph (networkx.Graph): The network topology
            propagation_delays (dict): frozenset({u, v}) -> delay in ms
            datarate_residuals (dict): frozenset({u, v}) -> residual bandwidth (MB/s)
        """
        self.graph = graph
        self.propagation = propagation_delays
        self.datarate = datarate_residuals

    def weight_fn(self, request):
        """
        Returns a weight function for use in shortest-path algorithms.
        Weight includes propagation delay and link availability.
        """
        def weight(u, v, d):
            edge_key = frozenset({u, v})
            if self.datarate[edge_key] < request.datarate:
                return np.inf  # prohibit use of saturated link
            return self.propagation[edge_key]

        return weight

    def shortest_paths(self, source, latency_cutoff, request):
        """
        Computes shortest paths from source to all reachable nodes
        under a latency cutoff using valid links only.

        Returns:
            lengths (dict): node -> cumulative delay
            routes (dict): node -> list of nodes in path
        """
        try:
            lengths, routes = nx.single_source_dijkstra(
                self.graph,
                source=source,
                weight=self.weight_fn(request),
                cutoff=latency_cutoff
            )
            return lengths, routes
        except Exception as e:
            return {}, {}

    def filter_valid_nodes(self, routes, request, compute_fn):
        """
        Filters valid placement nodes from route dictionary based on compute/memory.

        Parameters:
            routes (dict): node -> list of nodes in route
            request: service request containing vnf type and datarate
            compute_fn: function(node, vtype) -> (compute_demand, memory_demand)

        Returns:
            valid_routes (dict): node -> list of (u,v) edges
        """
        vtype = request.vtypes[len(request.vtypes_placed)]
        valid_routes = {}
        for node, path in routes.items():
            compute, memory = compute_fn(node, vtype)
            if compute <= self.graph.nodes[node]['compute'] and \
               memory <= self.graph.nodes[node]['memory']:
                valid_routes[node] = list(zip(path[:-1], path[1:]))
        return valid_routes
