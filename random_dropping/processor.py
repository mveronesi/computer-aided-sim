from simulator import BinBallSimulator
import numpy as np
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm


def simulate(n: int, d: int, seed: int) -> dict:
    simulator = BinBallSimulator(
        n=n,
        d=d,
        seed=seed
    )
    res = None
    simulator.execute()
    if simulator.bins is not None:
        res = {
            'max': np.max(simulator.bins),
            'min': np.min(simulator.bins),
            'avg': np.average(simulator.bins),
        }
    return res


def print_graph(
        x: list,
        max: list,
        min: list,
        avg: list,
        d: int) -> None:
    limit = lambda n, d: np.log(np.log(n))/np.log(d)
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.plot(x, max)
    ax.plot(x, min)
    ax.plot(x, avg)
    ax.plot(x, limit(x, d))
    ax.set_title(f'Output variables w.r.t. size of the problem\nLoad factor d={d}')
    ax.set_xlabel('Number of bins and balls (size of the problem)')
    ax.legend(['max', 'min', 'avg', 'max_limit'])


def main(args):
    args.n_max = int(args.n_max)
    n = np.arange(
        start=args.n_min,
        stop=args.n_max,
        step=args.n_step,
        dtype=np.int64
        )
    max_res = np.empty_like(n)
    min_res = np.empty_like(n)
    avg_res = np.empty_like(n)
    for i in tqdm(range(len(n))):
        res = simulate(
            n=n[i],
            d=args.d,
            seed=args.seed
            )
        max_res[i] = res['max']
        min_res[i] = res['min']
        avg_res[i] = res['avg']
    print_graph(
        x=n,
        max=max_res,
        min=min_res,
        avg=avg_res,
        d=args.d)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_max',
        help='Maximum number of balls and bins',
        type=float
    )
    parser.add_argument(
        '--n_min',
        help='Minimum number of balls and bins',
        default=50,
        type=int
    )
    parser.add_argument(
        '--n_step',
        help='Granularity of the samples',
        type=int,
        default=50
    )
    parser.add_argument(
        '-d',
        help='Load balancing parameter',
        type=int
    )
    parser.add_argument(
        '--seed',
        help='Random seed for the simulator',
        type=int,
        default=42
    )
    main(parser.parse_args())
