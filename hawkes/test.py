import numpy as np
np.random.seed(42)


def thinning():
    max_time = 100
    alpha = 2   # m parameter
    beta = 0.1  # lambda parameter
    sigma = lambda t: 20 if t <= 10 else 0
    T = { 0, }
    time = 0   # s: last point of the homogeneous process
    n = 0   # number of points in the non-homogeneous process
    tn = 0
    custom_exp = lambda t: beta*np.exp(-beta*(t-time))
    filter_T = lambda t: t if t-time <= 20 else None
    T = set(map(filter_T, T))
    T.discard(None)
    lam_bar = sigma(time) + alpha*sum(tuple(map(custom_exp, T)))
    while time < max_time:
        print(time)
        w = np.random.exponential(1/lam_bar)
        time = time + w
        T = set(map(filter_T, T))
        T.discard(None)
        lam_s = sigma(time) + alpha*sum(tuple(map(custom_exp, T)))
        u = np.random.uniform()
        if u < lam_s/lam_bar:
            n += 1
            tn = time
            T.add(tn)
        lam_bar = lam_s
    if tn > max_time:
        T.discard(tn)
    return T
