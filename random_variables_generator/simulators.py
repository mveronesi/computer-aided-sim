import numpy as np


class BinomialGenerator:
    def cdf(self, m: int) -> float:
        """
        IN: m integer value for which computing the binomial CDF
        OUT: the value of the CDF at point m
        """
        log_factorial = lambda n: 0 if n==0 else np.sum(
            np.log(np.arange(start=1, stop=n+1, dtype=int))
        )
        log_binomial = lambda n, k: log_factorial(n) - \
            log_factorial(k) - log_factorial(n-k)
        pdf = np.vectorize(
            lambda n, p, k: np.exp(log_binomial(n, k) + \
                k*np.log(p) + (n-k)*np.log(1-p))
            )
        return np.sum(pdf(self.n, self.p, np.arange(stop=m+1)))

    def convolution(self) -> int:
        """
        IN: None
        OUT: an integer number, i.e., an instance of the binomial
             random variable
        """
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
        """
        IN: None
        OUT: an integer number, i.e., an instance of the binomial
             random variable
        """
        u = self.generator.uniform()
        x = self.search(value=u)
        return x

    def geometric(self) -> int:
        """
        IN: None
        OUT: an integer number, i.e., an instance of the binomial
             random variable
        """
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

    def __init__(
            self, n: int,
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

    def __repr__(self) -> str:
        return f"""{self.method_str} generator with
n={self.n} and p={self.p}"""

    def __str__(self) -> str:
        return self.method_str


class TruncatedNormalGenerator:
    def __init__(
            self, mean: float,
            std_dev: float,
            approx_factor: int,
            verbose: bool,
            seed: int):
        """
        IN:
            - mean: a float value, the mean of the normal distribution
            - std_dev: a float value, the standard deviation of the
                    normal distribution.
            - approx_factor: integer value, the factor used for truncating
                    the tails of the normal distribution.
                    The suggested value is 5.
            - verbose: True if you want the generator to print warnings,
                    False otherwise.
            - seed: integer value, seed for the random generator.
        OUT: a fresh generator to create your normal random instances :)
        """
        self.mean = mean
        self.std_dev = std_dev
        self.variance = std_dev**2
        self.approx_factor = approx_factor
        self.verbose = verbose
        self.generator = np.random.default_rng(seed=seed)
        self.pdf = lambda x: np.exp(-(x-self.mean)**2/(2*self.variance))\
            / (np.sqrt(self.variance*2*np.pi))
        self.a = self.mean - self.approx_factor*self.std_dev
        self.b = self.mean + self.approx_factor*self.std_dev
        self.c = self.pdf(self.mean)
        self.P = 1/(self.c*(self.b-self.a))

    def generate(self, max_step: int=100) -> float:
        """
        IN: the number of maximum step for the acceptance/rejection technique.
        OUT: an instance of the truncated normal random variable.
        """
        i = 0
        generated = False
        x = None
        while not generated and i<max_step:
            x = self.generator.uniform(low=self.a, high=self.b)
            y = self.generator.uniform(low=0, high=self.c)
            fx = self.pdf(x)
            i += 1
            if y <= fx:
                generated = True
        if not generated and self.verbose:
            print(f'WARNING: {max_step} steps are not enough. \
            Try to increment max_step and/or to reduce the approx_factor.')
        return x
