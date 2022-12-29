from simulator import ThinningSimulator
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
from time import time


def main(args):
    simulator = ThinningSimulator(
        m=args.m,
        lam_exp=args.lam,
        h=args.h,
        end_time=args.end_time,
        active_thres_uni=args.active_threshold,
        seed=args.seed
    )
    start = time()
    infections = simulator.thinning()
    execution_time = time() - start
    print(f'The simulation took {execution_time} seconds.')
    cum_infections = np.cumsum(infections)
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(infections)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Infections')
    ax.set_title(f'Number of infections per day, h={args.h}')
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(cum_infections)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cumulative infections')
    ax.set_title(f'Cumulative number of infections, h={args.h}')
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
        help='Lambda parameter for h exponential.'
    )
    parser.add_argument(
        '--active_threshold',
        type=int,
        default=20,
        help='Maximum number of days for considering a subject infective, in the case of h uniform.'
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
