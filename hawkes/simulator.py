import numpy as np
from tqdm import tqdm


class ThinningSimulator:

    def get_active_points(self) -> np.ndarray:
        time = int(self.time)
        start = max(0, time - self.active_thres)
        return self.infected[start:time]

    def sum_h_exp(self) -> float:
        h_func_exp = lambda t, n: n*self.beta*np.exp(-self.beta*t)
        active_points = self.get_active_points()
        time = np.arange(
            start=len(active_points),
            stop=0,
            step=-1,
            dtype=int
            )
        return h_func_exp(time, active_points).sum()

    def sum_h_uni(self) -> float:
        active_points = self.get_active_points()
        total = active_points.sum()
        return 0.05*total

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
        self.infected = np.zeros(shape=(end_time,), dtype=int)
        self.T = {0,}
        self.active_T = {0,}
        if h == 'uniform':
            self.sum_h = self.sum_h_uni
        elif h == 'exponential':
            self.sum_h = self.sum_h_exp
        else:
            raise Exception(f'Function h={h} is not handled.')
    
    def thinning(self) -> np.ndarray:
        n = 0
        tn = 0
        self.time = 0
        gamma = lambda: self.sigma(self.time) + self.alpha*self.sum_h()
        gamma_bar = gamma()
        pbar = tqdm(desc='Thinning', total=self.end_time)
        while self.time < self.end_time and gamma_bar > 0:
            w = self.generator.exponential(1/gamma_bar)
            self.time = self.time + w
            pbar.update(w)
            gamma_s = gamma()
            u = self.generator.uniform()
            tn = int(self.time)
            if tn < self.end_time and \
                    u < gamma_s / gamma_bar:
                n += 1
                self.infected[tn] += 1
            gamma_bar = gamma_s
        return self.infected
