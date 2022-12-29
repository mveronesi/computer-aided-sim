from simulator import ThinningSimulator
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import atexit
from time import time


def main(args):
    simulator = ThinningSimulator(
        m=args.m,
        lam_exp=args.lam,
        h=args.h,
        end_time=args.end_time,
        active_thres_uni=args.active_threshold,
        death_rate=args.death_rate,
        interventions=args.interventions,
        seed=args.seed
    )
    start = time()
    infections, deaths = simulator.thinning()
    execution_time = time() - start
    print(f'The simulation took {execution_time} seconds.')
    cum_infections = np.cumsum(infections)
    cum_deaths = np.cumsum(deaths)
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].plot(infections)
    ax[0].set_xlabel('Time (days)')
    ax[0].set_ylabel('Infections')
    ax[0].set_title(f'Number of infections per day, h={args.h}')
    ax[1].plot(deaths)
    ax[1].set_xlabel('Time (days)')
    ax[1].set_ylabel('Deaths')
    ax[1].set_title(f'Number of deaths per day, h={args.h}')
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].plot(cum_infections)
    ax[0].set_xlabel('Time (days)')
    ax[0].set_ylabel('Cumulative infections')
    ax[0].set_title(f'Cumulative number of infections, h={args.h}')
    ax[1].plot(cum_deaths)
    ax[1].set_xlabel('Time (days)')
    ax[1].set_ylabel('Cumulative deaths')
    ax[1].set_title(f'Cumulative number of deaths per day, h={args.h}')
    plt.show()


if __name__ == '__main__':
    atexit.register(lambda: plt.close('all'))
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
        '--death_rate',
        type=float,
        default=0.02,
        help='Death rate of the desease.'
    )
    parser.add_argument(
        '--end_time',
        type=int,
        default=365,
        help='Maximum number of days for the simulation.'
    )
    parser.add_argument(
        '--interventions',
        action='store_true',
        help='Add interventions after day 20.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random simulator.'
    )
    main(parser.parse_args())
