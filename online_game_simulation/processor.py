from simulator import Simulator
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
FIG_SIZE = (17, 5)


def get_results(simulator) -> dict:
    avg_kill = 0
    for p in simulator.players:
        avg_kill += p.killed_players
        if p.alive == True:
            winner = p
    avg_kill /= simulator.n_players
    time_to_win = simulator.step / simulator.player_speed
    return {
        'time_to_win': time_to_win,
        'avg_kill': avg_kill,
        'winner_kill': winner.killed_players
    }


def simulation(arena_size, mobility_speed, number_initial_players, seed, verbose) -> dict:
    sim = Simulator(n_players=number_initial_players, arena_size=arena_size,\
        player_speed=mobility_speed, verbose=verbose, seed=seed)
    sim.execute()
    return get_results(sim)


def graphs_arena_size(start, end, step, mobility_speed, initial_players, seed, verbose) -> None:
    print('Plotting graphs for arena size variation')
    arena_size = np.arange(start=start, stop=end, step=step)
    time_to_win = np.empty_like(arena_size, dtype=float)
    winner_kill = np.empty_like(arena_size, dtype=int)
    avg_kill = np.empty_like(arena_size, dtype=float)
    for i in tqdm(range(len(arena_size))):
        results = simulation(arena_size=arena_size[i], mobility_speed=mobility_speed,\
            number_initial_players=initial_players, verbose=verbose, seed=seed)
        time_to_win[i] = results['time_to_win']
        winner_kill[i] = results['winner_kill']
        avg_kill[i] = results['avg_kill']
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=FIG_SIZE)
    ax[0].plot(arena_size, time_to_win)
    ax[0].set_title('Time to win w.r.t. size of arena')
    ax[0].set_xlabel('Side length (meters)')
    ax[0].set_ylabel('sec')
    ax[1].plot(arena_size, winner_kill)
    ax[1].set_title('Winner kill w.r.t. size of arena')
    ax[1].set_xlabel('Side length (meters)')
    ax[1].set_ylabel('no. of kill')
    ax[2].plot(arena_size, avg_kill)
    ax[2].set_title('Average kill w.r.t. size of arena')
    ax[2].set_xlabel('Side length (meters)')
    ax[2].set_ylabel('no. of kill')


def graphs_initial_players(start, end, step, arena_size, mobility_speed, seed, verbose) -> None:
    print('Plotting graphs for number of initial players variation')
    initial_players = np.arange(start=start, stop=end, step=step)
    time_to_win = np.empty_like(initial_players, dtype=float)
    winner_kill = np.empty_like(initial_players, dtype=int)
    avg_kill = np.empty_like(initial_players, dtype=float)
    for i in tqdm(range(len(initial_players))):
        results = simulation(arena_size=arena_size, mobility_speed=mobility_speed,\
            number_initial_players=initial_players[i], seed=seed, verbose=verbose)
        time_to_win[i] = results['time_to_win']
        winner_kill[i] = results['winner_kill']
        avg_kill[i] = results['avg_kill']
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=FIG_SIZE)
    ax[0].plot(initial_players, time_to_win)
    ax[0].set_title('Time to win w.r.t. initial players')
    ax[0].set_xlabel('#initial players')
    ax[0].set_ylabel('sec')
    ax[1].plot(initial_players, winner_kill)
    ax[1].set_title('Winner kill w.r.t. initial players')
    ax[1].set_xlabel('#initial players')
    ax[1].set_ylabel('no. of kill')
    ax[2].plot(initial_players, avg_kill)
    ax[2].set_title('Average kill w.r.t. initial players')
    ax[2].set_xlabel('#initial players')
    ax[2].set_ylabel('no. of kill')


def graph_mobility_speed(start, end, step, arena_size, initial_players, seed, verbose) -> None:
    print('Plotting graphs for mobility speed variation')
    mobility_speed = np.arange(start=start, stop=end, step=step)
    time_to_win = np.empty_like(mobility_speed, dtype=float)
    winner_kill = np.empty_like(mobility_speed, dtype=int)
    avg_kill = np.empty_like(mobility_speed, dtype=float)
    for i in tqdm(range(len(mobility_speed))):
        results = simulation(arena_size=arena_size, mobility_speed=mobility_speed[i],\
            number_initial_players=initial_players, seed=seed, verbose=verbose)
        time_to_win[i] = results['time_to_win']
        winner_kill[i] = results['winner_kill']
        avg_kill[i] = results['avg_kill']
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=FIG_SIZE)
    ax[0].plot(mobility_speed, time_to_win)
    ax[0].set_title('Time to win w.r.t. mobility speed')
    ax[0].set_xlabel('m/s')
    ax[0].set_ylabel('sec')
    ax[1].plot(mobility_speed, winner_kill)
    ax[1].set_title('Winner kill w.r.t. mobility speed')
    ax[1].set_xlabel('m/s')
    ax[1].set_ylabel('no. of kill')
    ax[2].plot(mobility_speed, avg_kill)
    ax[2].set_title('Average kill w.r.t. mobility speed')
    ax[2].set_xlabel('m/s')
    ax[2].set_ylabel('no. of kill')


def main(args):
    graphs_arena_size(
        start=args.start_arena_size,
        end=args.end_arena_size,
        step=args.step_arena_size,
        mobility_speed=args.mobility_speed,
        initial_players=args.initial_players,
        seed=args.seed,
        verbose=args.verbose
    )
    
    graphs_initial_players(
        start=args.start_initial_players,
        end=args.end_initial_players,
        step=args.step_initial_players,
        mobility_speed=args.mobility_speed,
        arena_size=args.arena_size,
        seed=args.seed,
        verbose=args.verbose
    )

    graph_mobility_speed(
        start=args.start_mobility_speed,
        end=args.end_mobility_speed,
        step=args.step_mobility_speed,
        arena_size=args.arena_size,
        initial_players=args.initial_players,
        seed=args.seed,
        verbose=args.verbose
    )
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena_size', type=int, default=50,\
        help='the default side length of the square arena (m)')
    parser.add_argument('--start_arena_size', type=int, default=10,\
        help='the starting size of the arena variation analysis')
    parser.add_argument('--end_arena_size', type=int, default=100,\
        help='the ending size of the arena variation analysis')
    parser.add_argument('--step_arena_size', type=int, default=10,\
        help='the granularity of the arena size variation analysis')
    parser.add_argument('--initial_players', type=int, default=5,\
        help='the default initial number of players')
    parser.add_argument('--start_initial_players', type=int, default=2,\
        help='the starting value for initial players variation analysis')
    parser.add_argument('--end_initial_players', type=int, default=20,\
        help='the ending value for initial players variation analysis')
    parser.add_argument('--step_initial_players', type=int, default=2,\
        help='the granularity of the initial players variation analysis')
    parser.add_argument('--mobility_speed', type=int, default=3,\
        help='the default mobility speed (m/s)')
    parser.add_argument('--start_mobility_speed', type=int, default=3,\
        help='the starting value for mobility_speed variation analysis')
    parser.add_argument('--end_mobility_speed', type=int, default=10,\
        help='the ending value for mobility_speed variation analysis')
    parser.add_argument('--step_mobility_speed', type=int, default=1,\
        help='the granularity of the mobility_speed variation analysis')
    parser.add_argument('--seed', type=int, default=42,\
        help='the seed for the simulator')
    parser.add_argument('--verbose', type=bool, default=False,\
        help='specify whether to print more informations during the simulation (FEATURE NOT COMPLETED)')
    args = parser.parse_args()
    main(args)
