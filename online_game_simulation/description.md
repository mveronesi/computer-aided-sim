This file reports the high-level description of the simulator developer in lab 3 (October 14, 2022)

# Input parameters

length of the side of the square (positive integer, representing meters)

speed of players (for this study fixed, positive integer, representing meters over a second m/s)

the number of players (positive integer greater than 1)



# Assumptions

fixed speed

discrete coordinates (a point in the simulated map represents a portion of the real map)

4 possible directions at each moment (up, down, left, right)

if a player goes out of the map, then it returns to the point from which it exited

all the initial position are randomly selected with uniform distribution, furthermore they are all different



# Player representation

Players are represented through the structure player, which contains the two coordinate values (non-negative integers), a boolean value which is true if and only if that player is still alive, the number of killed player (non-negative integer).



# Positions representation

The data for expressing the position of each player are stored in two main data structures:

a list L of size equals to the number of players. This structure contain elements of type Player (the one described in player representation)

a 2-D numpy array M of size (number_of_players)x(number_of_players), with data type integer. This data structure will contain -1 if a cell is empty, otherwise it will contain the index of the player in the list L described above



# Simulation algorithm

	begin

	alive_players = number_of_players

	while alive_players > 1

		for i=0 up to number_of_players-1

		    extract a direction from a uniform random variable

		    enter in M using the coordinates stored in L[i]

		    move the element in M

		    if the moved element exits from the boundaries bring it back to the old position

		        check whether the new position is already occupied

		            if it is and the index already in there is greater than the player which is moving

		                then substitute it without any other machinery (it position will be updated)

		            if it is and the index already in there is lower than the player which is moving

		                toss a coin and decide who the winner is

		                update the number of killed player in the winner

		                update the alive parameter of the looser

		apply a random shuffle to L

		count the number of still alive players

	end




The complexity of this algorithm is O(n), where n is the number of players in the game.