import pandas as pd
from os.path import sep


class BirthdayDistribution:
    """
    Class for handling a custom distribution file.
    It is simply a binding to a data source,
    it is useful to access the file only once.
    """
    def __init__(self):
        df = pd.read_csv(
            f'data{sep}estimated_realistic_distribution.csv')
        self.alphabet = df['day_number'].values
        self.probabilities = df['probability'].values
