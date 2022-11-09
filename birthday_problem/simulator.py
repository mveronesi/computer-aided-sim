import numpy as np
from birthday_distribution import BirthdayDistribution
from tqdm import tqdm


class ConflictSimulator:

    class UnhandledDistributionException(Exception):
        pass

    def __init__(
            self,
            m: int,             # the (fixed) number of people
            k: int,             # the number of experiments
            distribution: str,  # a str in ['uniform', 'realistic']
            seed: int,           # seed for the random generator
            verbose: bool = False
            ):
        """
        Setup the simulation parameters, which are:
        - m: the number of samples in the population (required only
             for running the second experiment)
        - k: the number of times to repeat the experiment
        - distribution: 'uniform' for using the uniform distribution,
                        'realistic' for using the estimated real distribution
                        (requires the distribution file to be in the data folder)
        - seed: the seed for the random generator
        - verbose: set to true if you want a progress bar during simulation
        """
        self.m = m
        self.k = k
        self.verbose = verbose
        self.generator = np.random.default_rng(seed=seed)
        if distribution == 'uniform':
            self.distribution = lambda size: self.generator.integers(
                low=1,
                high=365,
                endpoint=True,
                size=size
            )
        elif distribution == 'realistic':
            realistic_distribution = BirthdayDistribution()
            self.distribution = lambda size: self.generator.choice(
                a=realistic_distribution.alphabet,
                p=realistic_distribution.probabilities,
                size=size
            )
        else:
            raise self.UnhandledDistributionException\
                (f'{distribution} distribution is not\
                    handled by this class.')

    def exec_sim_1(self) -> float:
        self.ans_1 = np.empty(shape=(self.k,), dtype=int)
        for j in tqdm(range(self.k)) \
                if self.verbose else range(self.k):
            conflict = False
            S = set()
            i = 0
            while not conflict:
                x = self.distribution(size=None)
                if not conflict and x in S:
                    self.ans_1[j] = i
                    conflict = True
                S.add(x)
                i += 1
        return np.sum(self.ans_1) / self.k

    def exec_sim_2(self) -> float:
        L = self.distribution(size=(self.k, self.m, ))
        self.ans_2 = 0
        for i in tqdm(range(self.k)) \
                if self.verbose else range(self.k):
            S = np.unique(L[i, :])
            if len(S) < self.m:
                self.ans_2 += 1
        self.ans_2 /= self.k
        return self.ans_2
