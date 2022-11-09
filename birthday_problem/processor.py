from simulator import ConflictSimulator
from tqdm import tqdm
import argparse
import numpy as np
from collections.abc import Callable
from matplotlib import pyplot as plt


class SeedGenerator(Callable):
    """
    Auxiliary class for generating a series of random seed
    starting from a single seed. It requires the number of
    experiments k, and extract a seed from a uniform distribution
    in a space which has 3 more dimensions than the original one
    (so that it is more unlikely to peek the same seed twice).
    """
    def __init__(
            self,
            init_seed: int, # the seed for the generator
            k: int):        # the range for the uniform distribution
        self.k = k
        self.generator = np.random.default_rng(seed=init_seed)

    def __call__(self) -> int:
        return self.generator.integers(self.k, self.k**3)


def plot_distribution(samples: list) -> None:
    bins = np.arange(
        start=np.min(samples),
        stop=np.max(samples) + 1
    )
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.hist(x=samples, bins=bins,)
    ax.set_title('Number of times in which the first collision happend with a population size')
    ax.set_xlabel('Population size')
    ax.set_ylabel('Number of times in which the first collision happend')


def main(args):
    get_random_seed = SeedGenerator(
        init_seed=args.seed,
        k = max(args.k1, args.k2)
    )
    # Problem 1
    print('Problem 1')
    sim = ConflictSimulator(
        m=None, # it is not needed to fix the population size in the first problem
        k=args.k1,
        distribution=args.distribution,
        seed=get_random_seed(),
        verbose=True
    )
    avg_value = sim.exec_sim_1()
    print(f'The average size of samples to have a collision is: {int(np.ceil(avg_value))}')
    plot_distribution(sim.ans_1)
    # Problem 2
    print('Problem 2')
    m_values = np.arange(
        start=args.start,
        stop=args.stop+args.step,
        step=args.step,
        dtype=int
    )
    estimated_prob = np.empty_like(m_values, dtype=np.float64)
    limit = np.empty_like(m_values, dtype=np.float64)
    for i in tqdm(range(len(m_values))):
        m = m_values[i]
        sim = ConflictSimulator(
            m=m,
            k=args.k2,
            distribution=args.distribution,
            seed=get_random_seed()
        )
        estimated_prob[i] = sim.exec_sim_2()
        limit[i] = 1 - np.exp(-m**2/730)
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(m_values, estimated_prob)
    ax.plot(m_values, limit)
    ax.legend(
        ['estimated probability', 'theoretical limit'],
        loc='lower right'
        )
    ax.set_title('Probability of a collision w.r.t. size of the population')
    ax.set_xlabel('Size of the population [m]')
    ax.set_ylabel('Probability of birthday collision')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k1',
        type=int,
        help='The number of experiments for the first problem'
    )
    parser.add_argument(
        '--k2',
        type=int,
        help='The number of experiments for the second problem'
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
