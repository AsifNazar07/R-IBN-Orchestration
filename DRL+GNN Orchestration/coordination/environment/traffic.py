from typing import List, Dict, Tuple, Iterator
from functools import cmp_to_key

import numpy as np
import scipy.stats as stats
from numpy.random import default_rng, BitGenerator
from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson

# ----------- Basic Data Structure for Requests ------------ #

class Request:
    """A single service request with resource and delay requirements."""
    def __init__(self, arrival: float, duration: float, datarate: float,
                 max_latency: float, endpoints: Tuple[int, int], service: int):
        self.arrival = arrival
        self.duration = duration
        self.datarate = datarate
        self.max_latency = max_latency

        self.ingress, self.egress = map(int, endpoints)
        self.service = int(service)

        # Set by the environment
        self.vtypes: List[int] = None
        self.resd_lat: float = None

    def __str__(self) -> str:
        return f"Route: ({self.ingress}-{self.egress}); Duration: {round(self.duration, 2)}; " \
               f"Rate: {round(self.datarate, 2)}; Resd. Lat.: {round(self.resd_lat, 2)}; " \
               f"Lat.: {round(self.max_latency, 2)}; Service: {self.service}"

# ----------- Service Traffic Generator ------------ #

class ServiceTraffic:
    def __init__(self, rng: BitGenerator, service: int, horizon: float,
                 process: Dict, datarates: Dict, latencies: Dict,
                 endpoints: np.ndarray, rates: np.ndarray, spaths: Dict):
        self.rng = rng
        self.service = service
        self.horizon = horizon
        self.process = process
        self.datarates = datarates
        self.latencies = latencies
        self.endpoints = endpoints
        self.spaths = spaths

        # Max seed for reproducibility
        self.MAX_SEED = 2**30 - 1

        # Create time function for inhomogeneous Poisson process
        T = np.linspace(0.0, horizon - 1, horizon)
        self.rate_function = TimeFunction((T, np.ascontiguousarray(rates)))

    def sample_arrival(self, horizon: float) -> np.ndarray:
        """Sample arrival times based on inhomogeneous Poisson process."""
        seed = int(self.rng.integers(0, self.MAX_SEED))
        sim = SimuInhomogeneousPoisson([self.rate_function], end_time=horizon, verbose=False, seed=seed)
        sim.simulate()
        return sim.timestamps[0]

    def sample_duration(self, size: int) -> np.ndarray:
        return self.rng.exponential(scale=self.process['mduration'], size=size)

    def sample_datarates(self, size: int) -> np.ndarray:
        loc, scale = self.datarates['loc'], self.datarates['scale']
        a, b = self.datarates['a'], self.datarates['b']
        return stats.truncnorm.rvs((a - loc)/scale, (b - loc)/scale, loc, scale, size=size, random_state=self.rng)

    def sample_latencies(self, propagation: np.ndarray) -> np.ndarray:
        loc, scale = self.latencies['loc'], self.latencies['scale']
        a, b = self.latencies['a'], self.latencies['b']
        samples = stats.truncnorm.rvs((a - loc)/scale, (b - loc)/scale, loc, scale,
                                      size=propagation.size, random_state=self.rng)
        return samples * propagation

    def sample_endpoints(self, arrivals: np.ndarray) -> Tuple[List[int], List[int]]:
        ingresses, egresses = [], []
        for arr in arrivals:
            timestep = int(np.floor(arr))
            flatten = self.endpoints[timestep].ravel()
            index = np.arange(flatten.size)
            ingress, egress = np.unravel_index(self.rng.choice(index, p=flatten),
                                               self.endpoints[timestep].shape)
            ingresses.append(ingress)
            egresses.append(egress)
        return ingresses, egresses

    def sample(self) -> List[Request]:
        arrivals = self.sample_arrival(self.horizon)
        durations = self.sample_duration(len(arrivals))
        datarates = self.sample_datarates(len(arrivals))
        ingresses, egresses = self.sample_endpoints(arrivals)
        propagations = np.asarray([self.spaths[i][j] for i, j in zip(ingresses, egresses)])
        latencies = self.sample_latencies(propagations)

        return [
            Request(arr, dur, rate, lat, (ingr, egr), self.service)
            for arr, dur, rate, lat, ingr, egr in zip(arrivals, durations, datarates, latencies, ingresses, egresses)
        ]

# ----------- Combined Traffic Wrapper ------------ #

class Traffic:
    def __init__(self, processes: List[ServiceTraffic]):
        self.processes = processes

    def sample(self) -> List[Request]:
        all_requests = list(chain.from_iterable(proc.sample() for proc in self.processes))
        return sorted(all_requests, key=cmp_to_key(lambda r1, r2: r1.arrival - r2.arrival))

    def __iter__(self) -> Iterator[Request]:
        return iter(self.sample())

# ----------- Stub Traffic for Evaluation ------------ #

class TrafficStub:
    def __init__(self, trace: List[Request]):
        self.trace = trace

    def sample(self) -> List[Request]:
        return self.trace

    def __iter__(self) -> Iterator[Request]:
        return iter(self.trace)
