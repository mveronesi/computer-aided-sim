# Birthday problem simulator

To run the simulation, firstly recreate my virtual environment with the command

```pip install -r requirements.txt```

Secondly, take a look at the input parameters typing

```python processor.py --help```

and then execute accordingly.

Note that with "first problem" we intend the computation of the
average size of the population to have at least one collision,
while with "second problem" we mean the computation of the collision
probability against an arbitrary number of m values (population size).

Suggested values are:

```python processor.py --k1 100000 --k2 1000 --start 2 --stop 100 --step 1 --distribution realistic```
