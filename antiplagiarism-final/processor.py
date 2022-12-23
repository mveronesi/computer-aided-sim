from simulator import AntiPlagiarismSimulator
from pympler.asizeof import asizeof
import numpy as np
from argparse import ArgumentParser


class SeedGenerator:
    def __init__(self, size: int, seed: int):
        self.low = size
        self.high = size**3
        self.generator = np.random.default_rng(seed)

    def __call__(self) -> int:
        return self.generator.integers(
            low=self.low,
            high=self.high
        )


def order_of_magnitude(number: float) -> int:
    return int(np.log10(number))


def preliminary_study(simulator: AntiPlagiarismSimulator) \
     -> tuple[float, float]:
    """
    OUT:
        - the number of distinct sentences stored
        - the average size of each sentences in bytes
        - the size of the set containing the sentences
    """
    n_distinct_sentences = len(simulator.distinct_sentences)
    total_size = 0.0
    for sentence in simulator.distinct_sentences:
        total_size += asizeof(sentence)
    avg_size = total_size / n_distinct_sentences
    set_size = asizeof(simulator.set_distinct_sentences)
    return n_distinct_sentences, avg_size, set_size


def Bexp(simulator: AntiPlagiarismSimulator) -> int:
    """
    OUT:
        - Bexp: the number of bits for the fingerprint
                to observe no collisions.
    """
    for n_bits in range(30, 40):
        simulator.store_hash_sentences(hash_dim=int(np.exp2(n_bits)))
        if len(simulator.distinct_hash_sentences) == len(simulator.set_distinct_sentences):
            return n_bits
    return -1


def Bteo(p: float, m: int) -> int:
    """
    IN:
        - p: probability of observing a collision.
        - m: number of distinct sentences to be stored.
    OUT:
        - Bteo: the number of bits for observing
                a collision with probability p when
                there are n possible values
    """
    n = m**2 / (2*np.log(1/(1-p)))
    return int(np.ceil(np.log2(n)))

def false_positive_probability(
        simulator: AntiPlagiarismSimulator,
        bits: int) -> float:
    hash_dim = int(np.exp2(bits))
    simulator.store_hash_sentences(hash_dim=hash_dim)
    return len(simulator.distinct_hash_sentences) / hash_dim


def main(args):
    print('##### Inputs #####')
    print(f'The sentence size is {args.window_size}.')
    simulator = AntiPlagiarismSimulator(
        filepath=args.filepath,
        window_size=args.window_size
        )
    print('\n##### Preliminary study #####')
    distinct_sentences, avg_sentence_size, set_size = \
        preliminary_study(simulator)
    print(f'The number of distinct sentences is {distinct_sentences}.')
    print(f'The average sentence size is {avg_sentence_size:.2f} Bytes.')
    print('\n##### Set of sentences ######')
    print(f'The set which stores the sentences is {set_size/1024**2:.2f} MegaBytes.')
    print('\n##### Fingerprint set #####')
    bexp = Bexp(simulator=simulator)
    bteo = Bteo(
        p=0.5,
        m=len(simulator.distinct_sentences)
        )
    false_positive_prob = \
        simulator.false_positive_probability(
            hash_dim=int(np.exp2(bexp))
            )
    print(f'Bexp is {bexp} bits.')
    print(f'Bteo is {bteo} bits.')
    print(f'The false positive probability, when the bits used \
for the fingerprint is {bexp}, is in the order of magnitude \
of 10^{order_of_magnitude(false_positive_prob)}.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--filepath',
        type=str,
        default='/home/mveronesi/computer-aided-sim/antiplagiarism-final/commedia.txt',
        help='File source for the text used in the antiplagiarism system.'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=6,
        help='Fixed length of the sentences.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed.'
    )
    main(parser.parse_args())
