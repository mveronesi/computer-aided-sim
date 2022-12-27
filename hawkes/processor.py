from simulator import HawkesSimulator
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
from time import time


def main(args):
    simulator = HawkesSimulator(
        alpha=args.m,
        beta=args.lam,
        h=args.h,
        end_time=args.end_time,
        active_thres=args.active_threshold,
        seed=args.seed
    )
    start = time()
    points = simulator.thinning()
    execution_time = time() - start
    print(f'The simulation took {execution_time/60} minutes.')
    m = max(points)
    _, ax = plt.subplots(1, 1, figsize=(7,7,))
    ax.hist(x=points, bins=np.arange(stop=m+1, dtype=int))
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--h',
        type=str,
        default='exponential',
        help='String in ["uniform", "exponential"].'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=2,
        help='m parameter.'
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=0.1,
        help='Lambda parameter.'
    )
    parser.add_argument(
        '--active_threshold',
        type=int,
        default=20,
        help='Maximum number of days for considering a subject infective.'
    )
    parser.add_argument(
        '--end_time',
        type=int,
        default=100,
        help='Maximum number of days for the simulation.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random simulator.'
    )
    main(parser.parse_args())
