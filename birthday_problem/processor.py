from simulator import ConflictSimulator
from tqdm import tqdm
import argparse
import numpy as np
from collections.abc import Callable
from matplotlib import pyplot as plt


class SeedGenerator(Callable):
    def __init__(
            self,
            init_seed: int, # the seed for the generator
            k: int):        # the range for the uniform distribution
        self.generator = np.random.default_rng(seed=init_seed)
        self.k = k

    def __call__(self) -> int:
        return self.generator.integers(10, 10+self.k*100)
        # shifting values to avoid 0 as seed

def main(args):
    get_seed = SeedGenerator(
        init_seed=args.seed,
        k = args.repeat
    )
    m_values = np.arange(
        start=args.start,
        stop=args.stop,
        step=args.step,
        dtype=int
    )
    ans_1 = np.empty_like(m_values, dtype=np.float64)
    ans_2 = np.empty_like(m_values, dtype=np.float64)
    for i in tqdm(range(len(m_values))):
        m = m_values[i]
        sim = ConflictSimulator(
            m=m,
            k=args.repeat,
            distribution=args.distribution,
            seed=get_seed()
        )
        sim.exec()
        ans_1[i] = sim.ans_1
        ans_2[i] = sim.ans_2
    _, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 5,)
    )
    ax[0].plot(m_values, ans_1)
    ax[1].plot(m_values, ans_2)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--repeat',
        type=int,
        help='The number of experiments'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='The number of people in each experiment'
    )
    parser.add_argument(
        '--stop',
        type=int,
        help='The maximum number of people'
    )
    parser.add_argument(
        '--step',
        type=int,
        help='The step size for increasing the number of people'
    )
    parser.add_argument(
        '--distribution',
        type=str,
        default='uniform',
        help='The distribution for generating birthdays,\
            a string in ["uniform", "realistic"]'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The seed for the simulation'
    )
    main(parser.parse_args())
