import numpy as np
from pandas import Series
from typing import Callable


class Client:
    def __init__(
            self,
            type: str,
            arrival_time: float):
        self.type = type
        self.arrival_time = arrival_time

    def get_delay(
            self,
            time: float) -> float:
        return time - self.arrival_time


class QueueSimulator:

    class UnhandledServiceDistribution(Exception):
        pass

    def hyperexponential2(
            self,
            p: np.float128,
            mu1: np.float128,
            mu2: np.float128) -> np.float128:
        exp1 = lambda: self.generator.exponential(mu1)
        exp2 = lambda: self.generator.exponential(mu2)
        u = self.generator.uniform()
        return exp1() if u < p else exp2()

    def __init__(
            self,
            utilization: float,
            service_distribution: str,
            endtime: int,
            transient_batch_size: int,
            steady_batch_size: int,
            transient_tolerance: float,
            seed: int,
            verbose: bool):
        self.users = 0
        self.time = 0
        self.transient = True
        self.transient_batch_size = transient_batch_size
        self.steady_batch_size = steady_batch_size
        self.transient_tolerance = transient_tolerance
        self.endtime = endtime
        self.verbose = verbose
        self.utilization = utilization
        self.queue = list()
        self.fes = list()
        self.queue_sizes = list()
        self.delays = list()
        self.generator = np.random.default_rng(seed=seed)
        self.inter_arrival_distribution = lambda: \
            self.generator.exponential(1/self.utilization)
        if service_distribution == 'exp':
            self.service_distribution = lambda: \
                self.generator.exponential(1)
        elif service_distribution == 'det':
            self.service_distribution = lambda: 1
        elif service_distribution == 'hyp':
            self.service_distribution = lambda: \
                self.hyperexponential2(
                    p=0.9,
                    mu1=1-1/np.sqrt(2),
                    mu2=9/np.sqrt(2)+1
                )
        else:
            raise self.UnhandledServiceDistribution(
                f'{service_distribution} \
                    distribution is not implemented.'
                )
        # scheduling the first arrival
        self.create_event(
            time=self.inter_arrival_distribution(),
            action=self.arrival,
            name='arrival'
            )

    def create_event(
            self,
            time: float,
            action: Callable,
            name: str) -> None:
        self.fes.append({'time': time, 'action': action, 'name': name})

    def arrival(self) -> int:
        """
        - introduce a new user in the queue
        - schedule the next arrival
        - return the number of users in the queue
        """
        # schedule the next arrival
        self.create_event(
            time=self.time + self.inter_arrival_distribution(),
            action=self.arrival,
            name='arrival'
            )
        # the client arrives in the queue
        self.users += 1
        self.queue_sizes.append(self.users)
        self.queue.append(Client('type1', self.time))
        if (self.users == 1):
            # there wasn't any client in the queue
            # then we serve it immediately
            self.create_event(
                time=self.time + self.service_distribution(),
                action=self.departure,
                name='departure'
                )
        return self.users
              
    def departure(self) -> float:
        """
        - remove a user from the queue
        - if there was another user in the queue
           it schedules its departure
        - compute and return the delay of the removed user
        """
        removed: Client = self.queue.pop(0)
        delay = removed.get_delay(time=self.time)
        self.delays.append(delay)
        # remove the client from the queue only at departure time
        self.users -= 1
        if (self.users > 0):
            # if there was another client in the queue
            # then serve it
            self.create_event(
                time=self.time + self.service_distribution(),
                action=self.departure,
                name='departure'
                )
        return delay

    def collect_batch(self, collect: str) -> None:
        batch_size = self.transient_batch_size \
            if self.transient else self.steady_batch_size
        for _ in range(batch_size):
            self.fes.sort(key=lambda x: x['time'], reverse=True)
            next_event = self.fes.pop()
            # advance the simulator in time to execute the next event
            self.time = next_event['time']
            # execute the next event
            value = next_event['action']()
            if next_event['name'] == collect:
                self.means.append(value)

    def exec(self, collect_means: str) -> None:
        self.means = list()
        n = 0
        while self.time < self.endtime:
            self.collect_batch(collect=collect_means)
            n += 1
            self.cumulative_means = Series(data=self.means)\
                                    .expanding() \
                                    .mean()\
                                    .values
            if (len(self.cumulative_means) > 1):
                relative_diff = np.abs(self.cumulative_means[-1] -\
                    self.cumulative_means[-2]) / \
                        self.cumulative_means[-1]
                if self.transient \
                and relative_diff < self.transient_tolerance:
                    self.transient = False
                    self.transient_end = self.transient_batch_size*n
                    n = 0
