from simulator import QueueSimulator
import argparse
import numpy as np
from matplotlib import pyplot as plt


def main(args):
    sim = QueueSimulator(
        utilization=args.utilization,
        service_distribution=args.service_distribution,
        endtime=args.endtime,
        steady_batch_size=args.batch_size,
        transient_batch_size=5,
        transient_tolerance=1,
        seed=args.seed,
        verbose=args.verbose
    )
    transient_end = sim.exec()
    _, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 7)
        )
    ax[0].plot(sim.delays)
    ax[2].plot(sim.cumulative_means)
    ax[2].axvline(x=transient_end, color='r')
    ax[0].set_title('Delay')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Delay of a client at departure time')
    ax[1].plot(sim.queue_sizes)
    ax[1].set_title('Queue size')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Number of clients in the queue')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--utilization',
        type=float,
        help='Utilization factor of the server.'
    )
    parser.add_argument(
        '--service_distribution',
        type=str,
        help='A string in ["exp", "det", "hyp"], the distribution of the service time.'
    )
    parser.add_argument(
        '--endtime',
        type=int,
        help='The time of the simulation (default 1e5).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for batch means algorithm.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The seed for the random generator (default 42).'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Use this flag to print informations during the simulation.'
    )
    main(parser.parse_args())
