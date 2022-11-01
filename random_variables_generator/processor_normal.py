from simulators import TruncatedNormalGenerator
import argparse
from matplotlib import pyplot as plt
import time
import numpy as np
from tqdm import tqdm


def main(args):
    generator = TruncatedNormalGenerator(
        mean=args.mean,
        std_dev=args.std_dev,
        seed=args.seed,
        approx_factor=args.approx_factor,
        verbose=args.verbose
    )
    print(f'The generator has an acceptance probability of P={generator.P}')
    n = args.n_samples
    x = np.empty(shape=(n,), dtype=float)
    bins = np.arange(
        start=generator.a,
        stop=generator.b+args.discretization_factor,
        step=args.discretization_factor
        )
    start = time.time()
    for i in tqdm(range(n)):
        x[i] = generator.generate(max_step=args.max_step)
    print(f'Finished in {time.time() - start} seconds')
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.hist(x=x, bins=bins)
    ax.set_title(f'Truncated Normal Distribution\n\
        mean={generator.mean}, std_dev={generator.std_dev}\n\
        max_step={args.max_step}, approx_factor={generator.approx_factor}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mean',
        type=float,
        help='Mean of the normal distribution'
    )
    parser.add_argument(
        '--std_dev',
        type=float,
        help='Standard deviation of the normal distribution'
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        help='Use this flag to print info at generation time'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=100,
        help='Maximum steps for acceptance/rejection technique'
    )
    parser.add_argument(
        '--approx_factor',
        type=int,
        default=5,
        help='Approximation factor for truncating the tails\
            of the normal distribution (suggested 5)'
    )
    parser.add_argument(
        '--discretization_factor',
        type=float,
        default=0.1,
        help='Size of bins to plot the normal distribution approximation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random generator'
    )
    main(parser.parse_args())
