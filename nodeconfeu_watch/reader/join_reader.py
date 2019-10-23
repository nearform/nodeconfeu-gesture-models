
import numpy as np

from ._dataset import Dataset

def _join_datasets(datasets):
    x_all, y_all = zip(*datasets)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    return Dataset(x, y)

class JoinReader:
    def __init__(self, readers):
        self.classnames = readers[0].classnames

        self.all = _join_datasets([reader.all for reader in readers])
        self.train = _join_datasets([reader.train for reader in readers])
        self.validation = _join_datasets([reader.validation for reader in readers])
        self.test = _join_datasets([reader.test for reader in readers])