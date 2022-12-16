import numpy as np


class BinBallSimulator:
    def reset(self) -> None:
        """
        IN: None
        OUT: The simulator has been bringed to the initial
             conditions, i.e., a new random generator is
             built, the bins of a previous simulation are deleted
        """
        self.rnd_gen = np.random.default_rng(seed=self.seed)
        self.bins = None
        self.dtype = np.int64

    def __init__(self, n: int, d: int, seed: int):
        """
        IN:
        - an integer n, i.e., the number of bins and balls
        - an integer d, i.e., the load factor
        - the seed for the random generator
        OUT: a new object simulator
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.reset()
        
    def execute(self) -> None:
        """
        IN: None
        OUT: the simulation is executed accordingly to the
             parameters of the simulator, the bins now contains
             the balls and you can compute whatever statistic you want.
        """
        self.bins = np.zeros(
            shape=(self.n,),
            dtype=self.dtype
            )
        for _ in range(self.n):
            extracted_bins = self.rnd_gen.integers(
                self.n,
                size=self.d,
                dtype=self.dtype
                )
            min_bin = np.iinfo(self.dtype).max
            min_value = np.iinfo(self.dtype).max
            for i in extracted_bins:
                if self.bins[i] <= min_value:
                    min_bin = i
                    min_value = self.bins[i]
            self.bins[min_bin] += 1
