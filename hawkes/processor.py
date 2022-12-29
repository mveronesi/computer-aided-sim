from simulator import ThinningSimulator
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import atexit
from time import time
from scipy import stats


class SeedGenerator:
    def __init__(self, k: int, seed: int):
        self.generator = np.random.default_rng(seed)
        self.low = k
        self.high = k**3

    def __call__(self) -> int:
        return self.generator.integers(
            low=self.low,
            high=self.high
            )


def confidence_interval(
        values: np.ndarray,
        confidence: float
    ) -> tuple[float, float, float]:
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) / np.sqrt(n)
        if n < 30:
            return mean, *stats.t.interval(confidence, n-1, mean, std)
        else:
            return mean, *stats.norm(confidence, mean, std)


def main(args):
    k = args.k
    get_seed = SeedGenerator(k=k, seed=args.seed)
    total_infections = np.zeros(shape=(args.end_time, args.k,), dtype=np.int64)
    total_deaths = np.zeros_like(total_infections)
    total_cum_infections = np.zeros_like(total_infections)
    total_cum_deaths = np.zeros_like(total_infections)
    for i in range(k):
        print(f'Experiment n. {i}')
        simulator = ThinningSimulator(
            m=args.m,
            lam_exp=args.lam,
            h=args.h,
            end_time=args.end_time,
            active_thres_uni=args.active_threshold,
            death_rate=args.death_rate,
            interventions=args.interventions,
            seed=get_seed()
        )
        start = time()
        infections, deaths = simulator.thinning()
        execution_time = time() - start
        print(f'The simulation took {execution_time} seconds.')
        cum_infections = np.cumsum(infections)
        cum_deaths = np.cumsum(deaths)
        total_infections[:, i] = infections
        total_cum_infections[:, i] = cum_infections
        total_deaths[:, i] = deaths
        total_cum_deaths[:, i] = cum_deaths
    avg_infections = np.empty(shape=(args.end_time, 3,), dtype=np.float64)
    avg_deaths = np.empty_like(avg_infections)
    avg_cum_infections = np.empty_like(avg_infections)
    avg_cum_deaths = np.empty_like(avg_infections)
    for i in range(args.end_time):
        avg_infections[i, :] = confidence_interval(
            values=total_infections[i, :],
            confidence=args.confidence
            )
        avg_deaths[i, :] = confidence_interval(
            values=total_deaths[i, :],
            confidence=args.confidence
        )
        avg_cum_infections[i, :] = confidence_interval(
            values=total_cum_infections[i, :],
            confidence=args.confidence
        )
        avg_cum_deaths[i, :] = confidence_interval(
            values=total_cum_deaths[i, :],
            confidence=args.confidence
        )
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].fill_between(
        x=np.arange(args.end_time),
        y1=avg_infections[:, 1],
        y2=avg_infections[:, 2]
        )
    ax[0].plot(avg_infections[:, 0])
    ax[0].set_xlabel('Time (days)')
    ax[0].set_ylabel('Infections')
    ax[0].set_title(f'Number of infections per day, h={args.h}')
    ax[1].plot(avg_deaths)
    ax[1].set_xlabel('Time (days)')
    ax[1].set_ylabel('Deaths')
    ax[1].set_title(f'Number of deaths per day, h={args.h}')
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].plot(avg_cum_infections)
    ax[0].set_xlabel('Time (days)')
    ax[0].set_ylabel('Cumulative infections')
    ax[0].set_title(f'Cumulative number of infections, h={args.h}')
    ax[1].plot(avg_cum_deaths)
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
        default='uniform',
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
        default=50,
        help='Maximum number of days for the simulation.'
    )
    parser.add_argument(
        '--interventions',
        action='store_true',
        help='Add interventions after day 20.'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of times to repeat the experiments.'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Size of the confidence interval.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random simulator.'
    )
    main(parser.parse_args())
