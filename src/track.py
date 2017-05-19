import numpy as np


class Track:
    data = np.zeros((1))
    sample_rate = 0
    label = ''

    def __init__(self, (sample_rate, data), label):
        self.sample_rate = sample_rate
        self.data = data
        self.label = label
