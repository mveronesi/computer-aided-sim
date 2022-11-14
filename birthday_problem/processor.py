from simulator import ConflictSimulator
from tqdm import tqdm
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t, norm


class SeedGenerator:
    """
    Auxiliary class for generating a series of random seed
    starting from a single seed. It requires the number of
    experiments k, and extract a seed from a uniform distribution.
    """
    def __init__(
            self,
            seed: int, # the seed for the generator
            k: int):   # the range for the uniform distribution
        self.k = k
        self.generator = np.random.default_rng(seed=seed)

    def __call__(self) -> int:
        return self.generator.integers(self.k, self.k**3)


def main(args):
    get_random_seed = SeedGenerator(
        seed=args.seed,
        k = max(args.k1, args.k2)
    )
    # Problem 1
    print('Problem 1')
    sim = ConflictSimulator(
        m=None, # it is not needed to fix the population 
                # size in the first problem
        k=args.k1,
        distribution=args.distribution,
        seed=get_random_seed(),
        verbose=True
    )
    avg_value = sim.exec_sim_1()
    print(f'The average size of samples to have a collision is: {avg_value}')
    conf_int = norm.interval(
        0.99,
        avg_value,
        np.sqrt(np.var(sim.sim_1_samples)/args.k1)
    )
    print(f'The .99 confidence interval is: [{conf_int[0]}, {conf_int[1]}]')
    bins = np.arange(
        start=np.min(sim.sim_1_samples),
        stop=np.max(sim.sim_1_samples) + 1
    )
    indexes = np.digitize(x=sim.sim_1_samples, bins=bins)
    x = np.zeros_like(bins, dtype=int)
    for i in indexes:
        x[i] += 1
    x/=args.k1
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.bar(x)
    ax.set_title('Number of times in which the first collision happend with a population size')
    ax.set_xlabel('Population size')
    ax.set_ylabel('Number of times in which the first collision happend')
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
        sim.reset(
            m=m,
            k=args.k2,
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
        help='The number of people in each experiment in the second problem'
    )
    parser.add_argument(
        '--stop',
        type=int,
        help='The maximum number of people in each experiment in the second problem'
    )
    parser.add_argument(
        '--step',
        type=int,
        help='The step size for increasing the number of people in the second problem'
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
        help='The seed for the simulation (default 42)'
    )
    main(parser.parse_args())
