from simulators import BinomialGenerator
import argparse
from matplotlib import pyplot as plt
import time
import numpy as np
from tqdm import tqdm


def generate(
        generator: BinomialGenerator,
        n: int,
        k: int,
        skip: bool = False) -> None:
    if skip: return
    print(f'Using {str(generator)} generator...')
    x = np.arange(stop=n+1, dtype=int)
    y = np.zeros(shape=(n+1,), dtype=int)
    start = time.time()
    for _ in tqdm(range(k)):
        y[generator.generate()] += 1
    y = np.trim_zeros(filt=y, trim='b')
    x = x[:len(y)]
    print(f'{repr(generator)}')
    print(f'execution time: {time.time()-start}')
    _, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(7, 7,)
    )
    ax.bar(
        x=x,
        height=y,
        width=0.8
        )
    ax.set_title(repr(generator)+f'\nk={k}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability')


def main(args):
    args.n = int(args.n)
    conv_generator = BinomialGenerator(
        n=args.n,
        p=args.p,
        seed=args.seed,
        method='convolutional'
    )
    inv_generator = BinomialGenerator(
        n=args.n,
        p=args.p,
        seed=args.seed,
        method='inverse-transform'
    )
    geom_generator = BinomialGenerator(
        n=args.n,
        p=args.p,
        seed=args.seed,
        method='geometric'
    )
    generate(
        generator=conv_generator,
        n=args.n,
        k=args.k,
        skip=args.skip_conv
    )
    generate(
        generator=inv_generator,
        n=args.n,
        k=args.k,
        skip=args.skip_inv
    )
    generate(
        generator=geom_generator,
        n=args.n,
        k=args.k,
        skip=args.skip_geom
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        type=float,
        help="""Parameter n of the binomial
        random variable, i.e.,
        the number of experiments"""
    )
    parser.add_argument(
        '-p',
        type=float,
        help="""Parameter p of the binomial
            random variable, i.e.,
            the probability of a successfull
            experiment"""
    )
    parser.add_argument(
        '-k',
        type=int,
        default=1000,
        help='Number of instances to be generated'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random generator'
    )
    parser.add_argument(
        '--skip-conv',
        dest='skip_conv',
        action='store_true',
        help='Use this flag to not perform convolutional generation'
    )
    parser.add_argument(
        '--skip-inv',
        dest='skip_inv',
        action='store_true',
        help='Use this flag to not perform inverse-transform generation'
    )
    parser.add_argument(
        '--skip-geom',
        dest='skip_geom',
        action='store_true',
        help='Use this flag to not perform geometric generation'
    )
    main(parser.parse_args())
