import numpy as np


class ConflictSimulator:

    class UnhandledDistributionException(Exception):
        pass

    def __init__(
            self,
            m: int,             # the (fixed) number of people
            k: int,             # the number of experiments
            distribution: str,  # a str in ['uniform', 'realistic']
            seed: int           # seed for the random generator
            ):
        self.m = m
        self.k = k
        self.generator = np.random.default_rng(seed=seed)
        if distribution == 'uniform':
            self.distribution = lambda: self.generator.integers(
                low=1,
                high=365,
                endpoint=True
            )
        elif distribution == 'realistic':
            self.distribution = lambda: 0 # TO-DO
        else:
            raise self.UnhandledDistributionException\
                (f'{distribution} distribution is not\
                    handled by this class.')

    def exec(self) -> None:
        L = np.empty(shape=(self.k, self.m,), dtype=int)
        self.ans_1 = 0
        for i in range(self.k):
            conflict = False
            j = 0
            S = set()
            while (j < self.m or not conflict):
                x = self.distribution()
                if j < self.m:
                    L[i, j] = x
                if not conflict and x in S:
                    self.ans_1 += j # the conflict has been encountered
                    conflict = True
                S.add(x)
                j += 1
        self.ans_1 /= self.k
        self.ans_2 = 0
        for i in range(self.k):
            S = np.unique(L[i, :])
            if len(S) < len(L[i, :]):
                self.ans_2 += 1
        self.ans_2 /= self.k
