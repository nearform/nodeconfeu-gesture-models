
from collections import defaultdict
import numpy as np
import math
import os

train_size_ratio = 0.75
classnames = ['nothing', 'clap2', 'upup', 'swiperight', 'swipeleft']

for filename in os.listdir('./data/'):

    rng = np.random.RandomState(0)
    max_sequence_length = 0
    data_raw = defaultdict(list)

    with open(f'./data/{filename}') as fp:
        for line in fp:
            name_and_length, *values = map(str.strip, line.split(','))
            name = name_and_length.split('(')[0]

            data = np.fromiter(map(int, values),  'float32').reshape(-1, 3)
            data_raw[name].append(data)
            max_sequence_length = max(max_sequence_length, data.shape[0])

    x_train, y_train = [], []
    x_test, y_test = [], []

    for gesture_name, gesture_observations in data_raw.items():
        # pad sequences and create numeric target vector
        x_all = np.stack([
            np.pad(
                np.hstack([gesture, np.ones((gesture.shape[0], 1), dtype='float32')]),
                [(0, max_sequence_length - gesture.shape[0]), (0, 0)],
                'constant')
            for gesture in gesture_observations
        ])
        y_all = np.asarray([classnames.index(gesture_name)] * len(gesture_observations))
        # split data into train and test
        indexes = rng.permutation(len(gesture_observations))
        train_size = math.floor(len(gesture_observations) * train_size_ratio)

        x_train.append(x_all[indexes[0:train_size]])
        y_train.append(y_all[indexes[0:train_size]])
        x_test.append(x_all[indexes[train_size:]])
        y_test.append(y_all[indexes[train_size:]])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    np.savez(f'./data/{filename[0:-4]}.npz',
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        classnames=classnames
    )
