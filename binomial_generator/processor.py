from simulator import BinomialGenerator
import argparse
from matplotlib import pyplot as plt
import time
import numpy as np
from tqdm import tqdm
FIGSIZE=(5, 5,)


def generate(
        generator: BinomialGenerator,
        n: int,
        k: int) -> None:
    x = np.arange(stop=n+1, dtype=int)
    y = np.zeros(shape=(n+1,), dtype=int)
    start = time.time()
    for _ in tqdm(range(k)):
        y[generator.generate()] += 1
    print(f'{str(generator)}')
    print(f'execution time: {time.time()-start}')
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=FIGSIZE
    )
    ax.bar(
        x=x,
        height=y,
        width=0.8
        )
    ax.set_title(str(generator))
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
    print('Using convolutional generator...')
    generate(
        generator=conv_generator,
        n=args.n,
        k=args.k
    )
    print('Using inverse-transform generator...')
    generate(
        generator=inv_generator,
        n=args.n,
        k=args.k
    )
    print('Using geometric generator...')
    generate(
        generator=geom_generator,
        n=args.n,
        k=args.k
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
    main(parser.parse_args())
