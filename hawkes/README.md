# Hawkes process simulation through thinning algorithm

Re-create my virtual environment using

```pip install -r requirements.txt```

To see the list of all the available parameter run

```python processor.py --help```

To obtain the results in the assignment, just run the
code using the default parameters, i.e., run the command

```python processor.py```

Please note that all the results include a .99 confidence
interval obtained with 10 runs (using a CPU with 16 cores).
Since the speed at which the experiments complete strongly
depend on the number of virtual cores of your CPU,
feel free to lower the parameters ```--k``` to speed-up the execution.
Obviously, this will lead to results in wider confidence intervals. 
