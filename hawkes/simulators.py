import numpy as np


class HawkesSimulator:

    def __init__(
            self,
            max_time: int,
            h_func: str,
            death_rate: float,
            m: float,
            ancestors_max_age: float,
            seed: int):
        self.max_time = max_time
        self.time = 0
        self.m = m
        self.ancestors_max_age = ancestors_max_age
        self.generator = np.random.default_rng(seed=seed)
        if h_func == 'uniform':
            self.h = lambda t: 0.05 if t < 20 else 0
        elif h_func == 'exponential':
            self.h = lambda t: 0.1*np.exp(-0.1*t)
        else:
            raise Exception(f'Unhandled h={h_func}')
        self.death_rate = death_rate
        self.infected = np.zeros(shape=(max_time,), dtype=int)
        self.sigma = lambda t: self.generator.exponential(20) if t < 10 else 0
        self.active_ancestors = set()
        self.active_ancestors.add(0.0)
        self.infected[0] = 1

    def sum_h(self, time) -> float:
        result = 0.0
        active_ancestors = list(self.active_ancestors)
        for ancestor in active_ancestors:
            if time - ancestor < self.ancestors_max_age:
                result += self.h(time - ancestor)
            else:
                self.active_ancestors.remove(ancestor)
        return result

    def thinning(self):
        gamma = self.sigma(self.time) + self.m * self.sum_h(self.time)
        while self.time < self.max_time:
            time_delta = self.generator.exponential(gamma)
            next_time = self.time + time_delta
            gamma_new = self.sigma(next_time) + self.m * self.sum_h(next_time)
            p = gamma_new / gamma
            u = self.generator.uniform()
            if u < p:
                self.infected[int(next_time)] += 1
                self.active_ancestors.add(next_time)
                self.time = next_time
                gamma = gamma_new
            if p < 1e-6:
                print('ALLAHUAKBAR')
                print(self.time)
                print(gamma)
                print(gamma_new)
                exit(1)
