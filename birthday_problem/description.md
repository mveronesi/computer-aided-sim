# Input parameters
- m: the number of people
- k: the number of times to repeat the experiment

# Output parameters
- the average number of people to observe a conflict
- the probability that a conflict realizes within a group of m people

# Assumptions
- k, m are small enough so that a matrix k x m can fit into memory
- dates are represented as an integer number in {1..365}

# Main data structures
- a matrix L of size k x m, where each row is an experiment (i.e., we extract m birthdays k times)
- a set S for putting in the extracted dates

# Main algorithm
```begin
    input: k, m
    initialize the empty matrix L[1..k][1..m]
    ans_1 := 0
    for i:=1 up to k
        conflict := False
        j := 0
        initialize a new empty set S
        do while j < m or conflict is False
            x := a random instance generated accordingly to the uniform or realistic distribution
            if j <= m then
                L[i][j] := x
            if x is already present in S and conflict is False then
                ans_1 += j
                conflict := True
            add x to the set S
    ans_1 /= k  <-- this is the answer to the first question
    ans_2 := 0
    for i:=1 up to k
        if the row L[i][1..m] has a conflict in it then
            ans_2 += 1
     ans_2 /= k  <-- this is the answer to the second question
     output: ans_1, ans_2
end```

This algorithm is in the class of complexity O(m*k) (both in memory and time terms).

# Possible extensions
- Give an approximation of the distribution of the random variable describing the number of occurrences needed to have a conflict

