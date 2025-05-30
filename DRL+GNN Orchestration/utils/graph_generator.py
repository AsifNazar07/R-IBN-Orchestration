import os
import random
import logging
import argparse
from typing import Tuple

import networkx as nx
from geopy.distance import geodesic

SPEED_OF_LIGHT = 299792458  # m/s
PROPAGATION_FACTOR = 0.77  # optical fiber slowdown factor


def setup_logger():
    logger = logging.getLogger("graph_generator")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def compute_delay(src_coord, dst_coord) -> float:
    distance = geodesic(src_coord, dst_coord).meters
    delay_ms = (distance / SPEED_OF_LIGHT) * 1000 * PROPAGATION_FACTOR
    return round(delay_ms, 6)


def parse_graphml(file_path: str, compute_range: Tuple[float, float], mem_range: Tuple[float, float], bw_range: Tuple[float, float]) -> nx.Graph:
    if not file_path.endswith(".graphml"):
        raise ValueError(f"Invalid input file: {file_path} is not a .graphml file")

    logger.info(f"Reading GraphML file: {file_path}")
    raw_g = nx.read_graphml(file_path, node_type=int)
    G = nx.Graph()
    node_map = {}
    idx = 0

    for node, data in raw_g.nodes(data=True):
        if data.get("Internal", '0') != '1':
            continue
        node_map[node] = idx
        lat = float(data.get("Latitude"))
        lon = float(data.get("Longitude"))
        G.add_node(idx, compute=random.uniform(*compute_range),
                        memory=random.uniform(*mem_range),
                        pos=(lat, lon),
                        type="host")
        idx += 1

    logger.info(f"Added {len(G.nodes)} internal nodes")

    for u, v in raw_g.edges():
        if u not in node_map or v not in node_map:
            continue
        n1, n2 = node_map[u], node_map[v]
        coord1 = G.nodes[n1]['pos']
        coord2 = G.nodes[n2]['pos']
        delay = compute_delay(coord1, coord2)
        G.add_edge(n1, n2,
                   bandwidth=random.uniform(*bw_range),
                   datarate=random.uniform(*bw_range),
                   propagation=delay,
                   type="fiber")

    logger.info(f"Constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def save_graph(G: nx.Graph, path: str):
    logger.info(f"Saving graph to: {path}")
    nx.write_gpickle(G, path)


def main():
    parser = argparse.ArgumentParser(description="Generate GNN-compatible network graph from GraphML")
    parser.add_argument('--inputfile', type=str, required=True, help='Input GraphML file')
    parser.add_argument('--outputfile', type=str, default='./data/network.gpickle', help='Output gpickle path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compute', nargs=2, type=float, default=[0.1, 1.0], help='Range for compute')
    parser.add_argument('--memory', nargs=2, type=float, default=[0.1, 1.0], help='Range for memory')
    parser.add_argument('--bandwidth', nargs=2, type=float, default=[0.1, 1.0], help='Range for link bandwidth/datarate')

    args = parser.parse_args()
    random.seed(args.seed)

    G = parse_graphml(
        file_path=args.inputfile,
        compute_range=tuple(args.compute),
        mem_range=tuple(args.memory),
        bw_range=tuple(args.bandwidth)
    )
    save_graph(G, args.outputfile)


if __name__ == "__main__":
    main()
