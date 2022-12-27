import numpy as np
from time import time


class HawkesSimulator:

    def update_active_points(self) -> None:
        start = time()
        filter_T = lambda t: t if t-self.time <= self.active_thres\
             else None
        self.active_T = set(map(filter_T, self.active_T))
        self.active_T.discard(None)
        elapsed = time() - start
        print(f'Procedure completed in {elapsed} seconds.')

    def sum_h_exp(self) -> float:
        self.update_active_points()
        h_func_exp = lambda t: self.beta*np.exp(-self.beta*(t-self.time))
        return sum(tuple(map(h_func_exp, self.active_T)))

    def sum_h_uni(self) -> float:
        self.update_active_points()
        return 0.05*len(self.active_T)

    def __init__(
            self,
            alpha: int,
            beta: float,
            h: str,
            end_time: int,
            active_thres: int,
            seed: int):
        self.generator = np.random.default_rng(seed)
        self.alpha = alpha
        self.beta = beta
        self.active_thres = active_thres
        self.end_time = end_time
        self.sigma = lambda t: 20 if t <= 10 else 0
        self.T = {0,}
        self.active_T = {0,}
        if h == 'uniform':
            self.sum_h = self.sum_h_uni
        elif h == 'exponential':
            self.sum_h = self.sum_h_exp
        else:
            raise Exception(f'Function h={h} is not handled.')
    
    def thinning(self) -> set:
        n = 0
        tn = 0
        self.time = 0
        get_lam = lambda: self.sigma(self.time) + self.alpha*self.sum_h()
        lam_bar = get_lam()
        while self.time < self.end_time:
            w = self.generator.exponential(1/lam_bar)
            self.time = self.time + w
            lam_s = get_lam()
            u = self.generator.uniform()
            if u < lam_s / lam_bar:
                n += 1
                tn = self.time
                self.T.add(tn)
                self.active_T.add(tn)
            lam_bar = lam_s
        if tn > self.end_time:
            self.T.discard(tn)
        return self.T
