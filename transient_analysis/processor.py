from simulator import QueueSimulator
import argparse
from matplotlib import pyplot as plt


ARGS = {
    'utilization': 0.1,
    'service_distribution': 'exp',
    'endtime': 1000,
    'steady_batch_size': 100,
    'transient_batch_size': 10,
    'transient_tolerance': 1e-3,
    'seed': 42,
    'verbose': False
}


def main(args):
    sim = QueueSimulator(
        utilization=args['utilization'],
        service_distribution=args['service_distribution'],
        endtime=args['endtime'],
        steady_batch_size=args['steady_batch_size'],
        transient_batch_size=args['transient_batch_size'],
        transient_tolerance=args['transient_tolerance'],
        seed=args['seed'],
        verbose=args['verbose']
    )
    sim.exec(collect_means='departure')
    _, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 10)
        )
    ax[0].plot(sim.delays)
    ax[0].plot(sim.cumulative_means)
    ax[0].axvline(x=sim.transient_end, color='red')
    ax[0].set_title('Delay')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Delay of a client at departure time')
    
    ax[1].plot(sim.queue_sizes)
    ax[1].plot(sim.cumulative_means)
    ax[1].axvline(x=sim.transient_end, color='red')
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
        help='A string in ["exp", "det", "hyp"],\
            the distribution of the service time.'
    )
    parser.add_argument(
        '--endtime',
        type=int,
        help='The time of the simulation.'
    )
    parser.add_argument(
        '--steady_batch_size',
        type=int,
        default=100,
        help='Batch size for batch means algorithm.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The seed for the random generator.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Use this flag to print informations\
            during the simulation.'
    )
    main(ARGS)
