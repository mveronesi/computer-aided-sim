import numpy as np


class BinBallSimulator:
    def reset(self) -> None:
        self.rnd_gen = np.random.default_rng(seed=self.seed)
        self.bins = None
        self.dtype = np.int64

    def __init__(self, n: int, d: int, seed: int):
        self.n = n
        self.d = d
        self.seed = seed
        self.reset()
        
    def execute(self) -> None:
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
