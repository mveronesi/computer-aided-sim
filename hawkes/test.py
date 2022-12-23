import numpy as np

gen = np.random.default_rng(42)
time = 0.0
i = 0
while time<10.0:
    time += gen.exponential(1/20)
    i += 1
print(i)
