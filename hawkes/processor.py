from simulator import ThinningSimulator
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import atexit
from time import time
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pandas as pd
import os


class SeedGenerator:
    def __init__(self, k: int, seed: int):
        self.generator = np.random.default_rng(seed)
        self.low = k
        self.high = k**3

    def __call__(self, size) -> np.ndarray:
        return self.generator.integers(
            low=self.low,
            high=self.high,
            size=(size,)
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


def single_experiment(
        seed: int,
        args
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f'Experiment with seed={seed}')
    simulator = ThinningSimulator(
            m=args.m,
            lam_exp=args.lam_exp,
            h=args.h,
            end_time=args.end_time,
            lam_uni=args.lam_uni,
            death_rate=args.death_rate,
            interventions=args.interventions,
            intervention_factor=args.intervention_factor,
            seed=seed
        )
    infections, deaths = simulator.thinning()
    cum_infections = np.cumsum(infections)
    cum_deaths = np.cumsum(deaths)
    return infections, deaths, cum_infections, cum_deaths, simulator.rho_history

def plot(ax, x, y, confidence):
    ax.fill_between(
        x=x,
        y1=y[:, 1],
        y2=y[:, 2],
        color='lightblue'
        )
    ax.plot(
        y[:, 0],
        color='black',
        linestyle='dashed'
    )
    ax.legend((f'{confidence} confidence interval', 'value'))
    ax.set_xlabel('Time (days)')


def experiment(seed_generator: SeedGenerator, args, save: bool):
    total_infections = np.zeros(shape=(args.end_time, args.k,), dtype=np.int64)
    total_deaths = np.zeros_like(total_infections)
    total_cum_infections = np.zeros_like(total_infections)
    total_cum_deaths = np.zeros_like(total_infections)
    total_rho = np.zeros_like(total_infections)
    frozen_experiment = partial(single_experiment, args=args)
    start = time()
    with ThreadPoolExecutor() as executor:
        results = executor.map(frozen_experiment, seed_generator(args.k))
    elapsed_time = time() - start
    print(f'Done {args.k} experiments, took {elapsed_time} seconds, unpacking results...')
    for i, result in enumerate(results):
        total_infections[:, i] = result[0]
        total_cum_infections[:, i] = result[2]
        total_deaths[:, i] = result[1]
        total_cum_deaths[:, i] = result[3]
        total_rho[:, i] = result[4]
    total_rho = np.power(total_rho, 2)
    avg_infections = np.empty(shape=(args.end_time, 3,), dtype=np.float64)
    avg_deaths = np.empty_like(avg_infections)
    avg_cum_infections = np.empty_like(avg_infections)
    avg_cum_deaths = np.empty_like(avg_infections)
    avg_rho = np.empty_like(avg_infections)
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
        avg_rho[i, :] = confidence_interval(
            values=total_rho[i, :],
            confidence=args.confidence
        )
    times = np.arange(args.end_time)
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    plot(ax=ax[0], x=times, y=avg_infections, confidence=args.confidence)
    ax[0].set_ylabel('Infections')
    ax[0].set_title(f'Number of infections per day, h={args.h}')
    plot(ax=ax[1], x=times, y=avg_deaths, confidence=args.confidence)
    ax[1].set_ylabel('Deaths')
    ax[1].set_title(f'Number of deaths per day, h={args.h}')
    _, ax = plt.subplots(1, 2, figsize=(14,7))
    plot(ax=ax[0], x=times, y=avg_cum_infections, confidence=args.confidence)
    ax[0].set_ylabel('Cumulative infections')
    ax[0].set_title(f'Cumulative number of infections, h={args.h}')
    plot(ax=ax[1], x=times, y=avg_cum_deaths, confidence=args.confidence)
    ax[1].set_ylabel('Cumulative deaths')
    ax[1].set_title(f'Cumulative number of deaths per day, h={args.h}')
    output = dict()
    output['delta'] = args.intervention_factor
    if args.interventions:
        _, ax = plt.subplots(1, 1, figsize=(7, 7))
        plot(ax=ax, x=times, y=avg_rho, confidence=args.confidence)
        ax.plot(avg_rho[:, 0])
        ax.set_ylabel('Rho^2(t)')
        ax.set_xlabel('Time (days)')
        ax.set_title('Factor of interventions w.r.t. time')
        costs = total_rho.sum(axis=0)
        cost, cost_conf_int_left, cost_conf_int_right = \
            confidence_interval(
                values=costs,
                confidence=args.confidence
            )
        yearly_death = avg_cum_deaths[-1, 0]
        yearly_death_conf_int = tuple(avg_cum_deaths[-1, 1:3])
        print(f'The total cost of the interventions is {cost}')
        print(f'The total number of death is {yearly_death}')   
        output['cost'] = cost
        output['cost_conf_int_left'] = cost_conf_int_left
        output['cost_conf_int_right'] = cost_conf_int_right
        output['deaths'] = yearly_death
        output['deaths_conf_int_left'] = yearly_death_conf_int[0]
        output['deaths_conf_int_right'] = yearly_death_conf_int[1]
    if save:
        df = pd.DataFrame(output, index=[0])
        output_path = 'results.csv'
        df.to_csv(
            output_path,
            mode='a',
            header=not os.path.exists(output_path),
            index=False
            )


def search_delta(args):
    deltas = (20, 30, 32, 33, 34, 35, 40, 50,)
    generator = SeedGenerator(k=args.k, seed=args.seed)
    args.h = 'uniform'
    args.interventions = True
    args.end_time = 365
    for delta in deltas:
        print(f'Using delta={delta}')
        args.intervention_factor = delta
        experiment(seed_generator=generator, args=args, save=True)


def main(args):
    tmp = args.intervention_factor
    generator = SeedGenerator(k=args.k, seed=args.seed)
    print('Part 1')
    for h in ('uniform', 'exponential',):
        args.h = h
        args.interventions = False
        args.intervention_factor = 1
        args.end_time = 100
        experiment(seed_generator=generator, args=args, save=False)
        plt.show(block=False)
    print('Part 2')
    args.h = 'uniform'
    args.interventions = True
    args.intervention_factor = tmp
    args.end_time = 365
    experiment(seed_generator=generator, args=args, save=False)
    plt.show(block=False)
    input('Press Enter to close all the figures.')


if __name__ == '__main__':
    atexit.register(lambda: plt.close('all'))
    parser = ArgumentParser()
    parser.add_argument(
        '--m',
        type=int,
        default=2,
        help='m parameter.'
    )
    parser.add_argument(
        '--lam_exp',
        type=float,
        default=0.1,
        help='Lambda parameter for h exponential.'
    )
    parser.add_argument(
        '--lam_uni',
        type=float,
        default=0.05,
        help='Lambda parameter for h uniform.'
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
        '--k',
        type=int,
        default=15,
        help='Number of times to repeat the experiments.'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Size of the confidence interval.'
    )
    parser.add_argument(
        '--intervention_factor',
        type=int,
        default=32,
        help='Factor of the intervention, the lower the number, the higher are the interventions.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random simulator.'
    )
    args = parser.parse_args()
    main(args)
