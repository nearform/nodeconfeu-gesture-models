
import os
import re
import math
import itertools
import os.path as path
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd

extract_name_and_length = re.compile(r'^([A-Z0-9]+)\(([0-9]+)\),', flags=re.IGNORECASE)
Dataset = namedtuple('Dataset', ['x', 'y', 'person'])

def _join_dataset(datasets):
    x_all, y_all, person_all = zip(*datasets)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    person = np.concatenate(person_all)
    return Dataset(x, y, person)

def _normalize_to_filepath_list(files):
    normalized_files = defaultdict(list)

    for person_name, files_or_directories in files.items():
        for file_or_directory in files_or_directories:
            if path.basename(file_or_directory).startswith('.'):
                continue

            if path.isfile(file_or_directory):
                normalized_files[person_name].append(file_or_directory)
            elif path.isdir(file_or_directory):
                for filename in os.listdir(file_or_directory):
                    if filename.startswith('.'):
                        continue
                    filepath = path.join(file_or_directory, filename)
                    if path.isfile(filepath):
                        normalized_files[person_name].append(filepath)

    return normalized_files

class AccelerationReader:
    """Reads and splits acceleration data.

    Reads from pseudo-CSV files, described in the `files` arguments as
    a dict of { person: [files_or_directories] }. It will do a stratified
    split into train, validation, and test according to `validation_ratio`
    and `test_ratio`. The stratified split groups the observations by
    (person, gesture-name).

    Each CSV file should look like

       GestureName1(number_of_samples),x,y,z,x,y,z,...
       GestureName1(number_of_samples),x,y,z,x,y,z,...
       GestureName2(number_of_samples),x,y,z,x,y,z,...
       GestureName2(number_of_samples),x,y,z,x,y,z,...

    , where a sample is the (x,y,z) tuple.

    It is recommended to also specify the `classnames`, this is a list
    of gesture-names used to encode the target. Only observations where
    the gesture-name is included in `classnames` are included. If
    `classnames` is `None` they will be infered from the CSV files,
    but this requires an extra parsing phase.

    Each dataset are exposed as named tuples. `dataset.train`, `dataset.validation`,
    and `dataset.test`, which each have `x` and `y` properties.

    Arguments:
        files: Dict, describes the relative location of all CSV files.
        test_ratio: float, the ratio of test observations taken from each group
            of (person, gesture).
        validation_ratio: float, the ratio of validation observations taken from
            each group of (person, gesture).
        classnames: List[str], list of gesture-names as strings. Observatinos with
            gesture-names not included will be ignored from the CSV files.
        max_sequence_length: int, the max length of the sequences. If a sequence
            is not long enogth, it will be padded with zero.
        input_shape: either '1d' or '2d', if '2d' the
            `x.shape = (obs, length, 1, dims). This is useful when using the Conv2D
            kernel. If '1d' then `x.shape = (obs, length, dims).
        mask_dimention: bool, adds a 4th mask feature to the x-vector. The mask
            feature is 1 when a sample exists and 0 when for the padding.
        max_observaions_per_group: int, if a group is over represented that can cause
            issues, such as the model overlearning that group. Setting,
            `max_observaions_per_group` can help with that, as it limits the
            number of observations that a group can have.
        seed: int, the RNG-seed used to sample observations into train, validation,
            and test datasets.

    Attributes:
        train: Dataset(x, y), named tuple with the train data.
        validation: Dataset(x, y), named tuple with the validation data.
        test: Dataset(x, y), named tuple with the test data.
        classnames: List[str], list of classnames, can be used to decode the
            target/prediction labels.
        max_sequence_length: int, the length of sequences in the dataset,
            although.
    """

    def __init__(self, files,
                 test_ratio=0.1, validation_ratio=0.1,
                 classnames=None,
                 max_sequence_length=50,
                 input_shape='2d',
                 mask_dimention=False,
                 max_observaions_per_group=None,
                 seed=0):
        self.files = files
        self.seed = seed
        self.input_shape = input_shape
        self.output_dtype = output_dtype
        self.mask_dimention = mask_dimention
        self.max_observaions_per_group = max_observaions_per_group

        normalized_files = _normalize_to_filepath_list(files)

        if classnames is None or max_sequence_length is None:
            infered_classnames, infered_max_sequence_length = self._infer_dataset_properties(
                itertools.chain.from_iterable(normalized_files.values())
            )

        self.classnames = list(infered_classnames) \
            if classnames is None \
            else classnames

        self.max_sequence_length = infered_max_sequence_length \
            if max_sequence_length is None \
            else max_sequence_length

        self.people_names = list(normalized_files.keys())

        dataset_all = []
        for person_name, filepaths in normalized_files.items():
            parser = self._parse_csv(filepaths)
            dataset_all.append(
                self._as_numpy(parser, self.people_names.index(person_name))
            )

        self.all = _join_dataset(dataset_all)
        (self.train,
         self.validation,
         self.test) = self._stratified_split(self.all, test_ratio, validation_ratio)

    def details(self):
        """Returns a description of the datasets as a printable string."""

        files_details = []
        for person_name, files in self.files.items():
            files_details.append(f"    {person_name}: [{', '.join(files)}]")

        class_details = []
        class_labels = np.concatenate([self.train.y, self.validation.y, self.test.y])
        observations = class_labels.shape[0]
        for class_id in range(len(self.classnames)):
            class_name = self.classnames[class_id]
            class_count = np.sum(class_labels == class_id)
            ratio_str = "%.1f" % ((class_count / observations) * 100)
            class_details.append(f"    [{class_id}] {class_name}: {class_count} ({ratio_str}%)")

        return (
            f"Acceleration dataset:\n"
            f"  files:\n"
            f"{'\n'.join(files_details)}"
            f"\n"
            f"  properties:\n"
            f"    seed: {self.seed}\n"
            f"    input_shape: {self.input_shape}\n"
            f"    mask_dimention: {self.mask_dimention}\n"
            f"    max_sequence_length: {self.max_sequence_length}\n"
            f"    max_observaions_per_group: {self.max_observaions_per_group}\n"
            f"\n"
            f"  observations:\n"
            f"    validation: {self.validation.y.shape[0]}\n"
            f"    train: {self.train.y.shape[0]}\n"
            f"    test: {self.test.y.shape[0]}\n"
            f"\n"
            f"  classes:\n"
            f"{'\n'.join(class_details)}"
        )

    def savecsv(self, filepath, frequency=10):
        """Save all dataset as a single csv file in long-format

        The CSV file will have the columns:
            * subset
            * label
            * person
            * id
            * acceleration
            * time
            * dimension
        """
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
                person = pd.Series(data.person).map(
                    { person_id: person_name for person_id, person_name in enumerate(self.people_names) }
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
            id_vars=['subset', 'label', 'person', 'id'],
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
                        np.fromiter(map(int, values), dtype=np.float32).reshape(-1, 3),
                        np.asarray(self.classnames.index(name), dtype=np.int32)
                    )

    def _as_numpy(self, parser, person_index):
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

        person_numpy = np.full(y_numpy.shape[0], person_index)

        return Dataset(x_numpy, y_numpy, person_numpy)

    def _stratified_split(self, dataset, test_ratio=0, validation_ratio=0):
        train_x, train_y, train_person = [], [], []
        validation_x, validation_y, validation_person = [], [], []
        test_x, test_y, test_person = [], [], []

        # create stratified splits
        rng = np.random.RandomState(self.seed)
        for person_subset in range(len(self.people_names)):
            for y_subset in range(len(self.classnames)):
                # subset observations for this value
                x_subset = dataset.x[
                    np.logical_and(dataset.y == y_subset, dataset.person == person_subset),
                    ...
                ]
                num_obs = x_subset.shape[0]

                # There may be some combination of (person, y) that doesn't have any observations.
                if num_obs == 0:
                    continue

                if self.max_observaions_per_group and num_obs > self.max_observaions_per_group:
                    x_subset = x_subset[-self.max_observaions_per_group:, ...]
                    num_obs = self.max_observaions_per_group

                # calculate dataset sizes
                validation_size = math.ceil(num_obs * validation_ratio)
                test_size = math.ceil(num_obs * test_ratio)
                train_size = num_obs - (validation_size + test_size)

                if train_size <= 0:
                    raise ValueError(
                        f'too few observations of "{self.classnames[y_subset]}" to split dataset')

                # generate permutated indices
                indices = rng.permutation(num_obs)

                # append splits to final dataset
                validation_x.append(x_subset[indices[0:validation_size]])
                validation_y.append(np.full(validation_size, y_subset, dtype=dataset.y.dtype))
                validation_person.append(np.full(validation_size, person_subset, dtype=dataset.person.dtype))

                test_x.append(x_subset[indices[validation_size:validation_size + test_size]])
                test_y.append(np.full(test_size, y_subset, dtype=dataset.y.dtype))
                test_person.append(np.full(test_size, person_subset, dtype=dataset.person.dtype))

                train_x.append(x_subset[indices[validation_size + test_size:]])
                train_y.append(np.full(train_size, y_subset, dtype=dataset.y.dtype))
                train_person.append(np.full(train_size, person_subset, dtype=dataset.person.dtype))

        return (
            Dataset(np.concatenate(train_x), np.concatenate(train_y), np.concatenate(train_person)),
            Dataset(np.concatenate(validation_x), np.concatenate(validation_y), np.concatenate(validation_person)),
            Dataset(np.concatenate(test_x), np.concatenate(test_y), np.concatenate(test_person))
        )
