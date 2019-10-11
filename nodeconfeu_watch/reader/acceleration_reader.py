
import os
import re
import math
import os.path as path
import numpy as np
from collections import namedtuple

Dataset = namedtuple('Dataset', ['x', 'y'])

extract_name_and_length = re.compile(r'^([A-Z0-9]+)\(([0-9]+)\),', flags=re.IGNORECASE)

class AccelerationReader:
    def __init__(self, dirname, test_ratio=0.1, validation_ratio=0.1,
                 classnames=None,
                 max_sequence_length=None,
                 seed=0):
        self.dirname = dirname
        self.seed = seed

        filepaths = [
            path.join(dirname, filename) \
            for filename \
            in os.listdir(dirname) \
            if not filename.startswith('.')
        ]

        if classnames is None or max_sequence_length is None:
            infered_classnames, infered_max_sequence_length = self._infer_dataset_properties(filepaths)

        self.classnames = list(infered_classnames) \
            if classnames is None \
            else classnames

        self.max_sequence_length = infered_max_sequence_length \
            if max_sequence_length is None \
            else max_sequence_length

        parser = self._parse_csv(filepaths)
        x, y = self._as_numpy(parser)
        (
            self.train,
            self.validation,
            self.test
        ) = self._stratified_split(x, y, test_ratio, validation_ratio)

    def details(self):
        header = (
            f"Acceleration dataset:\n"
            f"  properties:\n"
            f"    seed: {self.seed}\n"
            f"    dirname: {self.dirname}\n"
            f"    max_sequence_length: {self.max_sequence_length}\n"
            f"\n"
            f"  observations:\n"
            f"    validation: {self.validation.y.shape[0]}\n"
            f"    train: {self.train.y.shape[0]}\n"
            f"    test: {self.test.y.shape[0]}\n"
            f"\n"
            f"  classes:\n"
        )

        class_details = []
        class_labels = np.concatenate([self.train.y, self.validation.y, self.test.y])
        observations = class_labels.shape[0]
        for class_id in range(len(self.classnames)):
            class_name = self.classnames[class_id]
            class_count = np.sum(class_labels == class_id)
            ratio_str = "%.1f" % ((class_count / observations) * 100)
            class_details.append(f"    [{class_id}] {class_name}: {class_count} ({ratio_str}%)")

        return (header + '\n'.join(class_details))

    def _infer_dataset_properties(self, filepaths):
        classnames = set()
        max_sequence_length = 0

        for filepath in filepaths:
            with open(filepath) as fp:
                for line in fp:
                    results = extract_name_and_length.match(line)
                    if results is None:
                        continue

                    classnames.add(results.group(1).lower())
                    max_sequence_length = max(max_sequence_length, int(results.group(2)))

        return (
            classnames,
            max_sequence_length
        )

    def _parse_csv(self, filepaths):
        for filepath in filepaths:
            with open(filepath) as fp:
                for line in fp:
                    results = extract_name_and_length.match(line)
                    if results is None:
                        continue

                    name = results.group(1).lower()
                    if name not in self.classnames:
                        continue

                    name_and_length, *values = map(str.strip, line.split(','))
                    yield (
                        np.fromiter(map(int, values), dtype='int8').reshape(-1, 3),
                        np.asarray(self.classnames.index(name), dtype='int32')
                    )

    def _as_numpy(self, parser):
        x_list, y_list = [], []
        for input_data, target_index in parser:
            x_list.append(input_data)
            y_list.append(target_index)

        x_numpy = np.stack([
            np.pad(
                np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)]),
                [(0, self.max_sequence_length - x.shape[0]), (0, 0)],
                'constant')
            for x in x_list
        ])
        y_numpy = np.asarray(y_list)

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
                raise ValueError(
                    f'too few observations of "{self.classnames[value]}" to split dataset')

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
