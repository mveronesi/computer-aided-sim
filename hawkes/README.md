# Hawkes process simulation through thinning algorithm

Re-create my virtual environment using

```pip install -r requirements.txt```

To have a list of all the available parameter type

```python processor.py --help```

To obtain the results required in the first part
of the assignment, use the following two commands

```python processor.py --h uniform```

```python processor.py --h exponential```

Please note that all the results include a .99 confidence
interval obtained with 15 runs.
Since the speed at which the experiments complete strongly
depend on the number of virtual threads of your CPU,
feel free to lower the parameters ```--k``` to speed-up the execution.
Obviously, this will result in wider intervals, but you can also
decrease ```--confidence``` to .95, in order to make them narrow again. 


