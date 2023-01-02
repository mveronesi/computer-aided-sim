import numpy as np
from tqdm import tqdm


class ThinningSimulator:

    def sum_h_exp(self) -> float:
        h_func_exp = lambda t, n: n*self.lam_exp*np.exp(-self.lam_exp*t)
        active_points = self.infected[:int(self.time)]
        time = np.arange(
            start=len(active_points),
            stop=0,
            step=-1,
            dtype=int
            )
        return h_func_exp(time, active_points).sum()

    def sum_h_uni(self) -> float:
        time = int(self.time)
        start = max(0, time - self.active_thres_uni)
        active_points = self.infected[start:time].sum()
        return self.lam_uni*active_points

    def rho(self) -> float:
        rho = 1
        if self.interventions and self.time >= 20:
            infections = self.infected[int(self.time)-1]
            deaths = int(np.ceil(self.death_rate*infections))
            rho = max(1, deaths / self.intervention_factor)
        tn = int(self.time)
        if tn < self.end_time:
            self.rho_history[tn] = rho
        return rho

    def __init__(
            self,
            m: int,
            lam_exp: float,
            h: str,
            end_time: int,
            lam_uni: float,
            death_rate: float,
            interventions: bool,
            intervention_factor: int,
            seed: int):
        self.generator = np.random.default_rng(seed)
        self.m = m
        self.seed = seed
        self.lam_exp = lam_exp
        self.lam_uni = lam_uni
        self.active_thres_uni = int(1/lam_uni)
        self.death_rate = death_rate
        self.interventions = interventions
        self.intervention_factor = intervention_factor
        self.end_time = end_time
        self.sigma = lambda t: 20 if t <= 10 else 0
        self.infected = np.zeros(shape=(end_time,), dtype=np.int64)
        self.rho_history = np.zeros(shape=(end_time,), dtype=float)
        if h == 'uniform':
            self.sum_h = self.sum_h_uni
        elif h == 'exponential':
            self.sum_h = self.sum_h_exp
        else:
            raise Exception(f'Function h={h} is not handled.')
    
    def thinning(self) -> tuple[np.ndarray, np.ndarray]:
        """
        OUT:
            - infections by day
            - deaths by day
        """
        self.time = 0.0
        gamma = lambda: self.sigma(self.time) + self.m/self.rho()*self.sum_h()
        gamma_bar = gamma()
        progress_bar = tqdm(desc=f'Thinning seed={self.seed}', total=self.end_time)
        while self.time < self.end_time and gamma_bar > 0:
            old_time = int(self.time)
            w = self.generator.exponential(1/gamma_bar)
            self.time = self.time + w
            tn = int(self.time)
            diff_time = tn - old_time
            if diff_time > 0:
                progress_bar.update(diff_time)
            gamma_s = gamma()
            u = self.generator.uniform()
            if tn < self.end_time and \
                    u < gamma_s / gamma_bar:
                self.infected[tn] += 1
            gamma_bar = gamma_s
        return self.infected, \
            np.ceil(self.death_rate*self.infected).astype(int)
