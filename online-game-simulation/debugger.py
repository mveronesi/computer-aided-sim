from simulator import Simulator

print('Starting debugger (endless simulation)')
sim = Simulator(n_players=1, arena_size=5, player_speed=1, verbose=True)
sim.reset()

try:
    while True:
        sim.print_arena()
        sim.print_players_list()
        sim.make_step()
        input('Press ENTER to continue the simulation\nCTRL+C to interrupt')
except KeyboardInterrupt:
    print('\nSimulation stopped\nBye!')
