import numpy as np
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
            u1: np.float128,
            u2: np.float128) -> np.float128:
        exp1 = lambda: self.generator.exponential(u1)
        exp2 = lambda: self.generator.exponential(u2)
        uniform = self.generator.uniform()
        return exp1() if uniform < p else exp2()

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
                    u1=1-1/np.sqrt(2),
                    u2=9/np.sqrt(2)+1
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

    def collect_batch(self, collect: str) -> float:
        batch_size = self.transient_batch_size \
            if self.transient else self.steady_batch_size
        total = 0
        count = 0
        while count < batch_size:
            self.fes.sort(key=lambda x: x['time'])
            next_event = self.fes.pop(0)
            # advance the simulator in time to execute the next event
            self.time = next_event['time']
            # execute the next event
            value = next_event['action']()
            if next_event['name'] == collect:
                total += value
                count += 1
        return total/count

    def exec(self) -> int:
        self.cumulative_means = list()
        old_mean = 0
        count = 0
        n = 0
        cumulative_mean = 0
        while (self.time < self.endtime):
            current_mean = self.collect_batch(collect='arrival')
            cumulative_mean = n/(n+1)*cumulative_mean + 1/(n+1)*current_mean
            self.cumulative_means.append(cumulative_mean)
            if count < 100 and self.transient:
                if abs(cumulative_mean-old_mean) < self.transient_tolerance:
                    count += 1
                else:
                    count = 0
            else:
                self.transient = False
            old_mean = cumulative_mean
        return count
