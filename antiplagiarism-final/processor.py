from simulator import AntiPlagiarismSimulator
from pympler.asizeof import asizeof
import numpy as np


def preliminary_study(simulator: AntiPlagiarismSimulator) \
     -> tuple[float, float]:
        """
        OUT:
            - the number of distinct sentences stored
            - the average size of each sentences in bytes
            - the size of the set containing the sentences
        """
        if simulator.processed is False:
            simulator.process()
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
    if simulator.processed is False:
        simulator.process()
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
    bteo = np.ceil(np.log2(n))
    return bteo


if __name__ == '__main__':
    simulator = AntiPlagiarismSimulator(
        filepath='commedia.txt',
        window_size=6
        )
    print(preliminary_study(simulator))
    print(Bexp(simulator))
    print(Bteo(p=0.5, m=len(simulator.distinct_sentences)))
