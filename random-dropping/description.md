Input:
- A positive integer number N, i.e., the size of the problem (N bins, N balls)
- An integer d>1, i.e., the random load balancing factor

Assumptions:
- N is small enough so that an array of shape (N, ) fits in the memory

Output:
- maximum number of balls in a bin
- minimum number of balls in a bin
- average number of balls in N bins

Data structure:
- an array B[1..N] of size N

Main algorithm:
begin
    input N>>0, d>1 integers
    initialize B[1..N] with zero values
    for i=1 up to N
        random pick d numbers in the interval [1..N]
        select the cell less occupied among those with an extracted index (from the previous point)
        increment by one the value contained in that cell
    compute the max, min, avg values of B
end

This algorithm executes in time O(d*N), with d a small constant, therefore O(N).

Possible extensions:
- Parallelize the for loop in the main algorithm: the less occupied cell will not be consistent,
  we should analyze the gain in terms of saved time w.r.t. the difference in the max occupied cell.
