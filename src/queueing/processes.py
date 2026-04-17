# queueing/processes.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ArrivalProcess:
    def sample_interarrival(self, rng: np.random.Generator, t: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class NoArrival(ArrivalProcess):
    def sample_interarrival(self, rng: np.random.Generator, t: float) -> float:
        return np.inf


@dataclass(frozen=True)
class PoissonArrival(ArrivalProcess):
    rate: float

    def sample_interarrival(self, rng: np.random.Generator, t: float) -> float:
        if self.rate <= 0:
            return np.inf
        return float(rng.exponential(1.0 / self.rate))


class ServiceProcess:
    def sample_service_time(self, rng: np.random.Generator, job, node: int, t: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ExponentialService(ServiceProcess):
    rate: float  # mu

    def sample_service_time(self, rng: np.random.Generator, job, node: int, t: float) -> float:
        if self.rate <= 0:
            raise ValueError("Service rate must be > 0")
        return float(rng.exponential(1.0 / self.rate))