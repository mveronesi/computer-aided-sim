from simulator import AntiPlagiarismSimulator
from pympler.asizeof import asizeof
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import atexit


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


def theoretical_false_positive(m: int, n: int, k: int) -> float:
    """
    IN:
        - m: number of distinct sentences
        - n: size of the bit array
        - k: number of hash functions (1 for bitstring)
    OUT:
        - p: probability of false positive
    """
    return (1 - np.exp(-k*m / (2*n)))**k


def k_opt(m: int, n: int) -> int:
    return int(np.ceil((n/m)*np.log(2)))


def optional_part(
        simulator: AntiPlagiarismSimulator,
        size: int,
        k: int,
        stop: int
        ) -> np.ndarray:
    bloom_filter = np.zeros(shape=(size,), dtype=bool)
    theoretical_size = lambda: -size/k * np.log(1-bloom_filter.sum()/size)
    theoretical_n_elements = []
    i = 0
    for sentence in simulator.distinct_sentences:
        for j in range(k):
            h = simulator.compute_hash(
                s=sentence,
                hash_dim=size,
                shift=j
                )
            bloom_filter[h] = True
        theoretical_n_elements.append(theoretical_size())
        i += 1
        if i >= stop: break
    theoretical_n_elements = np.array(theoretical_n_elements)
    absolute_error = np.abs(theoretical_n_elements - np.arange(start=1, stop=stop+1))
    return absolute_error
    

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
        false_positive_probability(
            simulator=simulator,
            bits=bexp
            )
    print(f'Bexp is {bexp} bits.')
    print(f'Bteo is {bteo} bits.')
    print(f'The false positive probability, when the bits used \
for the fingerprint is {bexp}, is in the order of magnitude \
of 10^{order_of_magnitude(false_positive_prob)}.')

    print('\n##### Bit string array #####')
    bits_exps = np.arange(start=19, stop=24, dtype=int)
    false_positive_prob = np.empty_like(bits_exps, dtype=float)
    theoretical_false_positive_prob = np.empty_like(bits_exps, dtype=float)
    for i in range(len(bits_exps)):
        p = false_positive_probability(
            simulator=simulator,
            bits=bits_exps[i]
            )
        false_positive_prob[i] = p
        theoretical_false_positive_prob[i] = theoretical_false_positive(
            m=len(simulator.distinct_sentences),
            n=int(np.exp2(bits_exps[i])),
            k=1
        )
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(
        bits_exps,
        theoretical_false_positive_prob,
        color='grey'
        )
    ax.scatter(
        bits_exps,
        false_positive_prob,
        marker='x',
        color='black'
        )
    ax.set_xlabel('Exponent of the bitstring size')
    ax.set_ylabel('Probability of false positive')
    ax.set_title('Bit string array')
    ax.set_xticks(bits_exps)
    ax.legend(
        ('Theoretical false positive probability',
        'Estimated false positive probability',)
        )

    print('\n##### Bloom filters #####')
    k_opts = np.empty_like(bits_exps, dtype=int)
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    for i, bit_exp in enumerate(bits_exps):
        k_opts[i] = k_opt(
            m=len(simulator.distinct_sentences),
            n=int(np.exp2(bit_exp))
        )
    ax.plot(bits_exps, k_opts, marker='.')
    ax.set_xticks(bits_exps)
    ax.set_xlabel('Exponent of the number of bits')
    ax.set_ylabel('k_opt')
    ax.set_title('Optimal number of k in function of the bit array size')
    pr_false_positive_teo = np.empty_like(bits_exps, dtype=float)
    pr_false_positive_exp = np.empty_like(bits_exps, dtype=float)
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    for i in range(len(bits_exps)):
        k = k_opts[i]
        b = bits_exps[i]
        size = int(np.exp2(b))
        pr_false_positive_teo[i] = theoretical_false_positive(
            m=len(simulator.distinct_sentences),
            n=size,
            k=k
        )
        bloom_filter = simulator.store_bloom_filter(
            k=k,
            size=size
        )
        pr_false_positive_exp[i] = (bloom_filter.sum() / size) ** k
    ax.plot(bits_exps, pr_false_positive_teo)
    ax.plot(bits_exps, pr_false_positive_exp)
    ax.set_yscale('log')
    ax.set_ylabel('Pr false positive')
    ax.set_xlabel('Exponent of the number of bits')
    ax.legend(('Theoretical', 'Experimental',))
    ax.set_title('Probability of false positive using the optimal number of hash functions')
    b = bits_exps[-1]
    k = k_opts[-1]
    size = int(np.exp2(b))
    stop = 20
    error = optional_part(simulator=simulator, size=size, k=k, stop=stop)
    n_elements_exp = np.arange(start=1, stop=stop+1)
    _, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(n_elements_exp, error, marker='o')
    ax.set_xticks(n_elements_exp)
    ax.set_xlabel('Number of stored elements')
    ax.set_ylabel('Absolute error')
    ax.set_title('Accuracy of the formula for computing the number of stored elements')
    plt.show(block=False)
    input('Press enter to close all the figures')


if __name__ == '__main__':
    atexit.register(lambda: plt.close('all'))
    parser = ArgumentParser()
    parser.add_argument(
        '--filepath',
        type=str,
        required=True,
        help='File source for the text used in the antiplagiarism system.'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=6,
        help='Fixed length of the sentences.'
    )
    main(parser.parse_args())
