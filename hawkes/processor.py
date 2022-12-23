from simulators import HawkesSimulator

sim = HawkesSimulator(max_time=100, h_func='uniform', death_rate=0.02, ancestors_max_age=10, seed=42, m=200)
sim.thinning()

