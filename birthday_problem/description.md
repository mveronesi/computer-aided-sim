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
Simulation for answering the first question:

```
begin
    input: k, m
    ans = 0
    repeat k times
        conflict = False
        initialize an empty set S
        i = 0
        while conflict is False
            generate an instance x of the random variable according to the given distribution (uniform or realistic)
            if conflict is False and x is not in S then
                ans += i
                conflict = True
            add x to the set S
            i++
    ans = ans / k
    output: ans
end
```

Simulation for answering the second question:
```
begin
    initialize L, a random matrix of size k x m in which
    each entry is an i.i.d. sample following the uniform or
    realistic distribution.
    ans = 0
    for i=1 up to k
        initialize an empty set S
        put all the values in L to S
        if the size of S is less than the size of L then there is a conflict
            ans++
    ans = ans/k
    output: ans
end
```

This algorithm is in the class of complexity O(m*k) (both in memory and time terms).

# Possible extensions
- Give an approximation of the distribution of the random variable describing the number of occurrences needed to have a conflict

