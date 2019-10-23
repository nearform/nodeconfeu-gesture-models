
import os
import re
import math
import itertools
import os.path as path

import numpy as np
import pandas as pd

from ._dataset import Dataset

extract_name_and_length = re.compile(r'^([A-Z0-9]+)\(([0-9]+)\),', flags=re.IGNORECASE)

class AccelerationReader:
    def __init__(self, dirname, test_ratio=0.1, validation_ratio=0.1,
                 classnames=None,
                 max_sequence_length=50,
                 input_dtype='float32',
                 input_shape='2d',
                 output_dtype='int32',
                 mask_dimention=True,
                 max_observaions_per_class=None,
                 seed=0):
        self.dirname = dirname
        self.seed = seed
        self.input_dtype = input_dtype
        self.input_shape = input_shape
        self.output_dtype = output_dtype
        self.mask_dimention = mask_dimention
        self.max_observaions_per_class = max_observaions_per_class

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
        self.all = self._as_numpy(parser)
        (
            self.train,
            self.validation,
            self.test
        ) = self._stratified_split(self.all.x, self.all.y, test_ratio, validation_ratio)

    def details(self):
        header = (
            f"Acceleration dataset:\n"
            f"  properties:\n"
            f"    seed: {self.seed}\n"
            f"    dirname: {self.dirname}\n"
            f"    input_shape: {self.input_shape}\n"
            f"    mask_dimention: {self.mask_dimention}\n"
            f"    max_sequence_length: {self.max_sequence_length}\n"
            f"    max_observaions_per_class: {self.max_observaions_per_class}\n"
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

    def savecsv(self, filepath, frequency=10):
        if not self.mask_dimention:
            raise ValueError('there must be a masked dimention to export to csv')

        df_all_list = []
        for subset in ['train', 'validation', 'test']:
            data = getattr(self, subset)

            x_nan_masked = data.x.reshape(data.x.shape[0], self.max_sequence_length, 4)
            x_nan_masked[x_nan_masked[:, :, 3] == 0, 0:3] = np.nan

            df_subset = pd.DataFrame(
                x_nan_masked.reshape(x_nan_masked.shape[0], -1),
                columns = itertools.chain.from_iterable((
                    [f'{i}.x', f'{i}.y', f'{i}.z', f'{i}.m'] \
                        for i in range(self.max_sequence_length)
                ))
            )
            df_subset = df_subset.assign(
                label = pd.Series(data.y).map(
                    { token_id: token for token_id, token in enumerate(self.classnames) }
                ),
                subset = subset
            )
            df_all_list.append(df_subset)

        df = pd.concat(df_all_list)
        df = df.assign(
            id = range(len(df))
        )

        # Convert wide format to long format
        df = pd.melt(
            df,
            id_vars=['subset', 'label', 'id'],
            var_name='time.dim',
            value_name='acceleration'
        )
        df_time_dim = df['time.dim'].str.split('.', expand=True)
        df = df.assign(
            time = pd.to_numeric(df_time_dim[0]) / frequency,
            dimension = df_time_dim[1]
        ).drop(['time.dim'], axis=1)

        # Remove mask dimension and nan values
        df = df.dropna(subset=['acceleration'])
        df = df[df['dimension'] != 'm']

        # Sort data for easy inspection
        df = df.sort_values(by=['id', 'time', 'dimension'])

        # Save dataframe
        df.to_csv(filepath, index=False)

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
                        np.fromiter(map(int, values), dtype=self.input_dtype).reshape(-1, 3),
                        np.asarray(self.classnames.index(name), dtype=self.output_dtype)
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

        if self.input_shape == '2d':
            x_numpy = x_numpy[:, :, np.newaxis, :]

        if not self.mask_dimention:
            x_numpy = x_numpy[..., 0:3]

        return Dataset(x_numpy, y_numpy)

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
            if self.max_observaions_per_class and num_obs > self.max_observaions_per_class:
                x_subset = x_subset[-self.max_observaions_per_class:, ...]
                num_obs = self.max_observaions_per_class

            # calculate dataset sizes
            validation_size = math.ceil(num_obs * validation_ratio)
            test_size = math.ceil(num_obs * test_ratio)
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
