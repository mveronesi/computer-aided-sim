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
    get_random_seed = SeedGenerator(
        init_seed=args.seed,
        k = args.repeat
    )
    m_values = np.arange(
        start=args.start,
        stop=args.stop,
        step=args.step,
        dtype=int
    )
    limit = np.empty_like(m_values, dtype=np.float64)
    ans_1 = np.empty_like(m_values, dtype=np.float64)
    ans_2 = np.empty_like(m_values, dtype=np.float64)
    for i in tqdm(range(len(m_values))):
        m = m_values[i]
        sim = ConflictSimulator(
            m=m,
            k=args.repeat,
            distribution=args.distribution,
            seed=get_random_seed()
        )
        sim.exec()
        ans_1[i] = sim.ans_1
        ans_2[i] = sim.ans_2
        limit[i] = 1-np.exp(-m**2/730)
    bins=np.arange(
            start=ans_1.min(),
            stop=ans_1.max()+1,
            dtype=float
            )
    indexes = np.digitize(ans_1, bins=bins,)
    y_ans_1 = np.zeros_like(bins, dtype=int)
    for i in indexes:
        y_ans_1[i] += 1
    _, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 15,)
    )
    ax[0].bar(
        x=bins,
        height=y_ans_1,
        width=0.4
        )
    ax[0].set_title('Histogram of the number of people needed to observe a conflict')
    ax[0].set_xlabel('Number of people [m]')
    ax[0].set_ylabel(f'Number of occurrences out of {args.repeat} experiments')
    ax[0].set_xticks(np.ceil(bins))
    ax[1].plot(m_values, ans_2)
    ax[1].plot(m_values, limit)
    ax[1].legend(
        labels=['Experiment results', 'Theoretical limit'],
        loc='lower right'
        )
    ax[1].set_title('Probability of birthday conflict w.r.t. the number of people')
    ax[1].set_xlabel('Number of people [m]')
    ax[1].set_ylabel('Probability of a birthday conflict')
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
