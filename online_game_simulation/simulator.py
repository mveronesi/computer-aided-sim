import numpy as np


class Player:
    def __init__(self, row, col, alive=True, killed_players=0) -> None:
        self.row = row
        self.col = col
        self.alive = alive
        self.killed_players = killed_players

    def __str__(self) -> str:
        return f'pos: {self.row}, {self.col}\nalive: {self.alive}\nkill: {self.killed_players}\n'


class Simulator:
    def __init__(self, n_players, arena_size, player_speed, verbose=False, seed=None) -> None:
        self.n_players = n_players
        self.arena_size = arena_size
        self.player_speed = player_speed
        self.verbose = verbose
        self.directions = (
            lambda player: (player.row-1 if player.row>0 else player.row, player.col, ), # up
            lambda player: (player.row+1 if player.row+1<self.arena_size else player.row, player.col, ), # down
            lambda player: (player.row, player.col-1 if player.col > 0 else player.col, ), # left
            lambda player: (player.row, player.col+1 if player.col+1<self.arena_size else player.col, ), # right
        )
        self.random_generator = np.random.default_rng(seed=seed)
        self.battle_rv = lambda: self.random_generator.integers(low=1, high=2, endpoint=True)
        self.direction_rv = lambda: self.random_generator.integers(len(self.directions))
    
    def generate_random_positions_matrix(self) -> list:
        res = np.empty(shape=(self.arena_size**2, ), dtype=(int, 2,))
        for i in range(self.arena_size):
            for j in range(self.arena_size):
                res[i*self.arena_size+j] = (i, j, )
        self.random_generator.shuffle(res)
        return res

    def reset(self) -> None:
        self.step = 0
        self.alive_players = self.n_players
        self.players = np.empty(shape=(self.n_players, ), dtype=Player)
        self.arena = -np.ones(shape=(self.arena_size, self.arena_size, ), dtype=int)
        positions = self.generate_random_positions_matrix()
        for i in range(self.n_players):
            pos = positions[i]
            self.players[i] = Player(row = pos[0], col=pos[1])
            self.arena[pos[0], pos[1]] = i

    def shuffle_players(self) -> None:
        self.random_generator.shuffle(self.players)
        self.arena = -np.ones(shape=(self.arena_size, self.arena_size, ), dtype=int)
        for i in range(len(self.players)):
            p = self.players[i]
            self.arena[p.row, p.col] = i

    def battle(self, player1, player2) -> None:
        coin = self.battle_rv()
        winner = player1 if coin==1 else player2
        loser = player1 if coin!=1 else player2
        self.arena[self.players[winner].row, self.players[winner].col] = winner
        self.players[winner].killed_players += 1
        self.players[loser].alive = False
        self.alive_players -= 1

    def move(self, player_index, next_pos) -> int:
        player = self.players[player_index]
        new_pos = next_pos(player)
        occupant = self.arena[new_pos[0], new_pos[1]]
        self.arena[player.row, player.col] = -1
        self.arena[new_pos[0], new_pos[1]] = player_index
        player.row = new_pos[0]
        player.col = new_pos[1]
        return occupant

    def make_step(self) -> None:
        if self.verbose == True:
            print(f'Step: {self.step}, alive players: {self.alive_players}')
        self.shuffle_players()
        for i in range(self.n_players):
            if self.players[i].alive == True:
                occupant = self.move(player_index=i, next_pos=self.directions[self.direction_rv()])
                if (occupant >= 0 and occupant < i):
                    self.battle(occupant, i)
        self.step += 1

    def execute(self) -> None:
        if self.verbose == True:
            print('Simulation start')
        self.reset()
        while self.alive_players > 1:
            self.make_step()
        if self.verbose == True:
            print('Simulation stop')

    def print_arena(self) -> None:
        for i in range(self.arena_size):
            for j in range(self.arena_size):
                val = self.arena[i, j]
                val = str(val) if val>=0 else '-'
                print(f'{val} ', end='')
            print()

    def print_players_list(self) -> None:
        for i in range(len(self.players)):
            p = self.players[i]
            print(f'Player {i}:\n{str(p)}')
