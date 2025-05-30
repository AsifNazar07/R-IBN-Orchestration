import heapq
import logging
from typing import List, Dict, Tuple
from itertools import islice, chain

import gym
import numpy as np
import networkx as nx
from gym import spaces
from munch import munchify
from tabulate import tabulate
from collections import Counter
from more_itertools import peekable

from coordination.environment.bidict import BiDict
from coordination.environment.traffic import Request, Traffic
from coordination.utils.state_encoder import GraphStateEncoder
from coordination.utils.reward import RewardStrategy
from coordination.utils.routing import RoutingStrategy
from coordination.utils.resources import ResourceManager


class ServiceCoordination(gym.Env):
    def __init__(self, net_path: str, process: Traffic, vnfs: List, services: List):
        self.net_path = net_path
        self.net = nx.read_gpickle(self.net_path)
        self.NUM_NODES = self.net.number_of_nodes()
        self.MAX_DEGREE = max([deg for _, deg in self.net.degree()])
        self.MAX_COMPUTE = self.net.graph['MAX_COMPUTE']
        self.MAX_LINKRATE = self.net.graph['MAX_LINKRATE']
        self.MAX_MEMORY = self.net.graph['MAX_MEMORY']
        self.HOPS_DIAMETER = self.net.graph['HOPS_DIAMETER']
        self.PROPAGATION_DIAMETER = self.net.graph['PROPAGATION_DIAMETER']

        self.process: Traffic = process
        self.vnfs: List[dict] = vnfs
        self.services: List[List[int]] = services
        self.NUM_SERVICES = len(self.services)
        self.MAX_SERVICE_LEN = max([len(service) for service in self.services])

        self.REJECT_ACTION = self.NUM_NODES + 1
        self.planning_mode = False

        self.ACTION_DIM = len(self.net.nodes) + 1
        self.action_space = spaces.Discrete(self.ACTION_DIM)
        self.OBS_SIZE = GraphStateEncoder.get_obs_size(self.net, services, vnfs)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float16)

        self.encoder = GraphStateEncoder(self.net, services, vnfs)
        self.rewarder = RewardStrategy(self.net)
        self.router = RoutingStrategy(self.net)
        self.resource_mgr = ResourceManager(self.net, vnfs)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

        self.pos = None
        self.info = None

        self.reset()

    def compute_state(self) -> np.ndarray:
        if self.planning_mode or self.done:
            return np.zeros(self.OBS_SIZE)
        return self.encoder.encode(self)

    def compute_reward(self, finalized: bool, deployed: bool, req: Request) -> float:
        return self.rewarder.compute(self, finalized, deployed, req)

    def compute_resources(self, node: int, vtype: int) -> Tuple[int]:
        return self.resource_mgr.compute_increment(self, node, vtype)

    def update_actions(self) -> None:
        self.valid_routes = self.router.get_valid_routes(self)

    def steer_traffic(self, route: List) -> None:
        self.router.steer(self, route)

    def step(self, action):
        rejected = (action == self.REJECT_ACTION)

        if not self.request in self.vtype_bidict.mirror:
            self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
            self.admission = {'deployed': False, 'finalized': False}

        if not (action in self.valid_routes or rejected):
            if self.planning_mode:
                raise RuntimeError('Invalid action taken.')
            self.done = True
            self.info[self.request.service].num_invalid += 1
            return self.compute_state(), 0.0, self.done, {}

        elif rejected:
            self.info[self.request.service].num_rejects += 1
            self.admission = {'deployed': False, 'finalized': True}
            self.release(self.request)
            reward = self.compute_reward(True, False, self.request)
            self.done = self.progress_time()
            return self.compute_state(), reward, self.done, {}

        final_placement = self.place_vnf(action)
        if final_placement:
            try:
                _, route = nx.single_source_dijkstra(
                    self.net, source=action, target=self.request.egress, weight=self.get_weights, cutoff=self.request.resd_lat)
                route = ServiceCoordination.get_edges(route)
                self.steer_traffic(route)
                exit_time = self.request.arrival + self.request.duration
                heapq.heappush(self.deployed, (exit_time, self.request))
                self.update_info()
                self.admission = {'deployed': True, 'finalized': True}
                reward = self.compute_reward(True, True, self.request)
                self.done = self.progress_time()
                return self.compute_state(), reward, self.done, {}

            except nx.NetworkXNoPath:
                self.info[self.request.service].no_egress_route += 1
                self.admission = {'deployed': False, 'finalized': True}
                self.release(self.request)
                reward = self.compute_reward(True, False, self.request)
                self.done = self.progress_time()
                return self.compute_state(), reward, self.done, {}

        reward = self.compute_reward(False, False, self.request)
        self.update_actions()

        if not self.valid_routes:
            self.info[self.request.service].no_extension += 1
            self.admission = {'deployed': False, 'finalized': True}
            self.release(self.request)
            self.done = self.progress_time()
            return self.compute_state(), reward, self.done, {}

        return self.compute_state(), reward, self.done, {}

    def place_vnf(self, node: int) -> bool:
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        compute, memory = self.compute_resources(node, vtype)
        self.computing[node] -= compute
        self.memory[node] -= memory

        self.occupied = {key: self.occupied[key] + delta for key, delta in
                         zip(['compute', 'memory', 'datarate'], [compute, memory, 0.0])}

        route = self.valid_routes[node]
        self.steer_traffic(route)
        self.vtype_bidict[(node, vtype)] = self.request

        if len(self.vtype_bidict.mirror[self.request]) >= len(self.request.vtypes):
            return True
        return False

    def get_weights(self, u: int, v: int, d: Dict) -> float:
        return self.router.get_weight(self, u, v)

    def replace_process(self, process):
        self.process = process
        self.reset()

    def reset(self) -> np.ndarray:
        self.trace = peekable(iter(self.process))
        self.request = next(self.trace)
        self.request.resd_lat = self.request.max_latency
        self.request.vtypes = self.services[self.request.service]
        self.done = False
        self.time = self.request.arrival
        self.num_requests = 1

        KEYS = ['accepts', 'requests', 'skipped_on_arrival', 'no_egress_route', 'no_extension', 'num_rejects',
                'num_invalid', 'cum_service_length', 'cum_route_hops', 'cum_compute',
                'cum_memory', 'cum_datarate', 'cum_max_latency', 'cum_resd_latency']
        self.info = [munchify(dict.fromkeys(KEYS, 0.0)) for _ in range(len(self.services))]
        self.info[self.request.service].requests += 1

        self.computing = {node: data['compute'] for node, data in self.net.nodes(data=True)}
        self.memory = {node: data['memory'] for node, data in self.net.nodes(data=True)}
        self.datarate = {frozenset({src, trg}): data['datarate'] for src, trg, data in self.net.edges(data=True)}
        self.propagation = {frozenset({src, trg}): data['propagation'] for src, trg, data in self.net.edges(data=True)}

        self.deployed, self.valid_routes = [], {}

        self.vtype_bidict = BiDict(None, val_btype=list)
        self.vtype_bidict = BiDict(self.vtype_bidict, val_btype=list)

        self.routes_bidict = BiDict(None, val_btype=list, key_map=lambda key: frozenset(key))
        self.routes_bidict = BiDict(self.routes_bidict, val_btype=list)

        self.routes_bidict[self.request] = (None, self.request.ingress)
        self.update_actions()

        if not self.valid_routes:
            self.progress_time()

        return self.compute_state()

    def progress_time(self) -> bool:
        while self.trace:
            self.request = next(self.trace)
            self.routes_bidict[self.request] = (None, self.request.ingress)
            self.info[self.request.service].requests += 1
            self.request.resd_lat = self.request.max_latency
            self.request.vtypes = self.services[self.request.service]
            self.time += self.request.arrival - self.time
            self.num_requests += 1

            while self.deployed and self.deployed[0][0] < self.time:
                _, service = heapq.heappop(self.deployed)
                self.release(service)

            self.update_actions()
            if self.valid_routes:
                return False

            self.info[self.request.service].skipped_on_arrival += 1

        return True

    def release(self, req: Request) -> None:
        if req not in self.vtype_bidict.mirror:
            return

        serving_vnfs = Counter(self.vtype_bidict.mirror[req])
        for (node, vtype), count in serving_vnfs.items():
            config = self.vnfs[vtype]
            supplied_rate = sum([r.datarate for r in self.vtype_bidict[(node, vtype)]])
            updated_rate = supplied_rate - count * req.datarate
            prev_cdem, prev_mdem = self.score(supplied_rate, config)
            after_cdem, after_mdem = self.score(updated_rate, config)
            self.computing[node] += prev_cdem - after_cdem
            self.memory[node] += prev_mdem - after_mdem

        del self.vtype_bidict.mirror[req]
        route = self.routes_bidict.pop(req, [])
        for src, trg in route[1:]:
            self.datarate[frozenset({src, trg})] += req.datarate

    def update_info(self) -> None:
        service = self.request.service
        self.info[service].accepts += 1
        self.info[service].cum_service_length += len(self.request.vtypes)
        self.info[service].cum_route_hops += len(self.routes_bidict[self.request])
        self.info[service].cum_datarate += self.request.datarate
        self.info[service].cum_max_latency += self.request.max_latency
        self.info[service].cum_resd_latency += self.request.resd_lat

    @staticmethod
    def get_edges(nodes: List) -> List:
        return list(zip(islice(nodes, 0, None), islice(nodes, 1, None)))

    @staticmethod
    def score(rate, config):
        if rate <= 0.0:
            return (0.0, 0.0)
        elif rate > config['max. req_transf_rate']:
            return (np.inf, np.inf)

        rate = rate / config['scale']
        compute = config['coff'] + config['ccoef_1'] * rate + config['ccoef_2'] * (rate ** 2) + \
                  config['ccoef_3'] * (rate ** 3) + config['ccoef_4'] * (rate ** 4)
        memory = config['moff'] + config['mcoef_1'] * rate + config['mcoef_2'] * (rate ** 2) + \
                 config['mcoef_3'] * (rate ** 3) + config['mcoef_3'] * (rate ** 4)

        return (max(0.0, compute), max(0.0, memory))
