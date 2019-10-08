
import math
import numpy as np
from collections import namedtuple

Dataset = namedtuple('Dataset', ['x', 'y'])

class AccelerationDataset:
    def __init__(self, filepath, test_ratio=0.1, validation_ratio=0.1,
                 classnames=['nothing', 'clap2', 'upup', 'swiperight', 'swipeleft'],
                 seed=0):
        self.classnames = classnames
        self.seed = seed

        parser = self._parse_csv(filepath)
        x, y = self._as_numpy(parser)
        self.train, self.validation, self.test = self._stratified_split(x, y, test_ratio, validation_ratio)

    def _parse_csv(self, filepath):
        with open(filepath) as fp:
            for line in fp:
                name_and_length, *values = map(str.strip, line.split(','))
                name = name_and_length.split('(')[0]

                yield (
                    np.fromiter(map(int, values), 'int8').reshape(-1, 3),
                    self.classnames.index(name)
                )

    def _as_numpy(self, parser):
        max_sequence_length = 0
        x_list, y_list = [], []
        for input_data, target_index in parser:
            max_sequence_length = max(max_sequence_length, input_data.shape[0])
            x_list.append(input_data)
            y_list.append(target_index)

        x_numpy = np.stack([
            np.pad(
                np.hstack([x, np.ones((x.shape[0], 1), dtype='float32')]),
                [(0, max_sequence_length - x.shape[0]), (0, 0)],
                'constant')
            for x in x_list
        ])
        y_numpy = np.asarray(y_list, 'int32')

        return (x_numpy, y_numpy)

    def _stratified_split(self, x, y, test_ratio=0, validation_ratio=0):
        train_x, train_y = [], []
        validation_x, validation_y = [], []
        test_x, test_y = [], []

        # create stratified splits
        rng = np.random.RandomState(self.seed)
        for value in range(len(self.classnames)):
            # subset observations for this value
            x_subset = x[y == value, ...]
            num_obs = x_subset.shape[0]

            # calculate dataset sizes
            validation_size = math.ceil(num_obs * validation_ratio)
            test_size = math.ceil(num_obs * validation_ratio)
            train_size = num_obs - (validation_size + test_size)

            if train_size <= 0:
                raise ValueError('zero data remaining for training dataset')

            # generate permutated indices
            indices = rng.permutation(num_obs)

            # append splits to final dataset
            validation_x.append(x_subset[indices[0:validation_size]])
            validation_y.append(np.full(validation_size, value, dtype=y.dtype))

            test_x.append(x_subset[indices[validation_size:validation_size + test_size]])
            test_y.append(np.full(test_size, value, dtype=y.dtype))

            train_x.append(x_subset[indices[validation_size + test_size:]])
            train_y.append(np.full(train_size, value, dtype=y.dtype))

        return (
            Dataset(np.concatenate(train_x), np.concatenate(train_y)),
            Dataset(np.concatenate(validation_x), np.concatenate(validation_y)),
            Dataset(np.concatenate(test_x), np.concatenate(test_y))
        )
