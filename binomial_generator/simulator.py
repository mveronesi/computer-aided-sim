import numpy as np


class BinomialGenerator:
    def cdf(self, m: int) -> float:
        log_factorial = lambda n: 0 if n==0 else np.sum(np.array(
            [np.log(i) for i in range(1, n+1)]
            ))
        log_binomial = lambda n, k: log_factorial(n) - \
            log_factorial(k) - log_factorial(n-k)
        pdf = np.vectorize(
            lambda n, p, k: np.exp(log_binomial(n, k) + \
                k*np.log(p) + (n-k)*np.log(1-p))
            )
        return np.sum(pdf(self.n, self.p, np.arange(stop=m+1)))

    def convolution(self) -> int:
        counter = np.vectorize(
            lambda x: 1 if x < self.p else 0
            )
        return np.sum(counter(
            self.generator.uniform(size=self.n)))

    def search(self,
            value: float,
            lower: int = 1,
            upper: int = 100) -> int:
        """
        IN:
            - value is a float in (0, 1)
            - lower is the lower bound in the cdf domain
            - upper is a truncation of the upper bound in the cdf domain
        OUT:
            - return an integer x such that F(x) <= value < F(x+1)
        """
        prev_cdf = 0
        for x in range(lower, upper+1):
            next_cdf = self.cdf(x)
            if prev_cdf <= value < next_cdf:
                return x
            prev_cdf = next_cdf
        return upper+1
        
    def inverse_transform(self) -> int:
        u = self.generator.uniform()
        x = self.search(value=u)
        return x

    def geometric(self) -> int:
        m = 0
        q = 0
        while q <= self.n:
            u = self.generator.uniform()
            g = np.ceil(np.log(u)/np.log(1-self.p))
            q += g
            m += 1
        return m-1

    METHODS = {
        'convolutional': convolution,
        'inverse-transform': inverse_transform,
        'geometric': geometric
    }

    def __init__(self, n: int,
            p: int,
            method: str,
            seed: int = None):
        """
        IN:
            - n, p parameters of the binomial random variable
            - method: a string in ['convolutional',
            'inverse-transform', 'geometric']
        """
        self.n = n
        self.p = p
        self.method = self.METHODS[method]
        self.method_str = method
        self.generator = np.random.default_rng(seed=seed)
    
    def generate(self) -> int:
        return self.method(self)

    def __str__(self) -> str:
        return f"""{self.method_str} generator with
n={self.n} and p={self.p}"""
