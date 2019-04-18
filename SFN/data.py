import os
from copy import deepcopy
from typing import List, Callable, Tuple
from collections import namedtuple

import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from termcolor import cprint
from tqdm import trange

from reject.ReusableObject import ReusableObject


def dtype_to_name(dtype: str):
    return dtype.lower().split("_")[-1]


def name_to_train_and_val_num(name: str) -> tuple:
    if name == "mnist":
        return 55000, 5000
    else:
        raise ValueError


def get_to_one_hot(num_class: int) -> Callable:
    lb = LabelBinarizer()
    lb.fit(range(num_class))
    return lb.transform


def preprocess_x(name: str, xs: List[np.ndarray]) -> tuple:
    if name == "mnist":
        return tuple(x / 255 for x in xs)
    else:
        raise ValueError


def get_tfds(dtype: str, data_dir: str, to_2d: bool):

    name = dtype_to_name(dtype)
    assert name in tfds.list_builders()

    data_dir = data_dir or os.path.join("..", "{}_data".format(name.upper()))  # e.g. ../MNIST_data

    # https://www.tensorflow.org/datasets/datasets
    loaded, info = tfds.load(
        name=name,
        split=["train", "test"],
        data_dir=data_dir,
        batch_size=-1,
        with_info=True,
    )

    # Get numpy matrix
    train_and_validation, test = tfds.as_numpy(loaded)

    # Reshaping
    _, w, h, _ = train_and_validation["image"].shape
    train_and_validation_x = train_and_validation["image"]
    if to_2d:
        train_and_validation_x = train_and_validation_x.reshape(-1, w * h)
        test_x = test["image"].reshape(-1, w * h)
    else:
        train_and_validation_x = train_and_validation_x.reshape(-1, w, h)
        test_x = test["image"].reshape(-1, w, h)

    # Preprocess
    train_and_validation_x, test_x = preprocess_x(name, [train_and_validation_x, test_x])

    # Training Validation Separation
    # this is necessary because tfds does not support validation separation.
    train_num = name_to_train_and_val_num(name)[0]
    train_x = train_and_validation_x[:train_num]
    val_x = train_and_validation_x[train_num:]

    # One hot labeling
    to_one_hot = get_to_one_hot(info.features["label"].num_classes)
    DataLabel = namedtuple("DataLabel", "train_labels, validation_labels, test_labels")
    data_label = DataLabel(
        train_labels=to_one_hot(train_and_validation["label"][:train_num]),
        validation_labels=to_one_hot(train_and_validation["label"][train_num:]),
        test_labels=to_one_hot(test["label"]),
    )
    return data_label, train_x, val_x, test_x


def get_permuted_datasets(dtype: str, n_tasks: int, data_dir=None, base_seed=42, to_2d=True) -> tuple:

    data_label, train_x, val_x, test_x = get_tfds(dtype, data_dir, to_2d)

    # Pixel Permuting
    task_permutation = []
    for task in range(n_tasks):
        np.random.seed(task + base_seed)
        task_permutation.append(np.random.permutation(train_x.shape[-1]))

    _train_xs, _val_xs, _test_xs = [], [], []
    for task in range(n_tasks):
        _train_xs.append(train_x[:, task_permutation[task]])
        _val_xs.append(val_x[:, task_permutation[task]])
        _test_xs.append(test_x[:, task_permutation[task]])

    return data_label, _train_xs, _val_xs, _test_xs


def sample_indices(sz, ratio):
    return np.random.choice(np.arange(sz, dtype=int), size=int(sz * ratio), replace=False)


def update_distance_k_center(dists, xs, current_id):
    dists = deepcopy(dists)
    for i in range(xs.shape[0]):
        current_dist = np.linalg.norm(xs[i, :] - xs[current_id, :])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists


def get_k_center_indices(xs: np.ndarray, sz: int):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    if xs.shape[0] == sz:
        return list(range(sz))

    current_id = 0
    indices = [current_id]

    dists = update_distance_k_center(np.full(xs.shape[0], np.inf), xs, current_id)
    for _ in trange(1, sz):
        current_id = int(np.argmax(dists))
        dists = update_distance_k_center(dists, xs, current_id)
        indices.append(current_id)

    return indices


def slice_xs(xs, indices):
    return list(map(lambda ndarr: ndarr[indices], xs))


class PermutedCoreset(ReusableObject):

    def __init__(self, data_labels, train_xs, val_xs, test_xs,
                 sampling_ratio: float or List[float],
                 sampling_type: str = None,
                 seed: int = 42,
                 load_file_name: str = None):

        if load_file_name and self.load(load_file_name):
            self.loaded = True
            return
        else:
            self.loaded = False

        self.sampling_type = sampling_type or "uniform"

        train_sz, val_sz, test_sz = tuple(map(lambda l: len(l[0]), [train_xs, val_xs, test_xs]))
        if isinstance(sampling_ratio, float):
            self.sampling_ratio_list = [sampling_ratio] * 3
        elif isinstance(sampling_ratio, list):
            self.sampling_ratio_list = sampling_ratio
        else:
            raise TypeError

        np.random.seed(seed)

        if self.sampling_type == "uniform":
            train_sampled_idx, val_sampled_idx, test_sampled_idx = tuple(map(
                sample_indices,
                [train_sz, val_sz, test_sz],
                self.sampling_ratio_list,
            ))

        elif self.sampling_type == "k-center":
            train_sampled_idx, val_sampled_idx, test_sampled_idx = tuple(map(
                get_k_center_indices,
                [train_xs[0], val_xs[0], test_xs[0]],
                [int(sz * ratio) for sz, ratio in zip([train_sz, val_sz, test_sz], self.sampling_ratio_list)]
            ))

        else:
            raise ValueError

        self.train_xs = slice_xs(train_xs, train_sampled_idx)
        self.val_xs = slice_xs(val_xs, val_sampled_idx)
        self.test_xs = slice_xs(test_xs, test_sampled_idx)

        self.train_labels = data_labels.train_labels[train_sampled_idx]
        self.val_labels = data_labels.validation_labels[val_sampled_idx]
        self.test_labels = data_labels.test_labels[test_sampled_idx]

        self.num_tasks = len(self.train_xs)

        cprint("""{name} initialized\
                  \n - num_tasks: {num_tasks}\
                  \n - train: {train_len} ({train_ratio})\
                  \n - validation: {val_len} ({val_ratio})\
                  \n - test: {test_len} ({test_ratio}) \n""".format(**{
            "name": self.__class__.__name__,
            "num_tasks": self.num_tasks,
            "train_len": len(train_sampled_idx),
            "val_len": len(val_sampled_idx),
            "test_len": len(test_sampled_idx),
            "train_ratio": self.sampling_ratio_list[0],
            "val_ratio": self.sampling_ratio_list[1],
            "test_ratio": self.sampling_ratio_list[2],
        }), "blue")

    def __getitem__(self, item):
        return self.train_xs[item], self.train_labels, \
               self.val_xs[item], self.val_labels, \
               self.test_xs[item], self.test_labels

    def reduce_xs(self, small_sz: int):
        assert small_sz <= len(self.train_labels)
        small_indices = np.arange(small_sz)
        self.train_xs = slice_xs(self.train_xs, small_indices)
        self.train_labels = self.train_labels[small_indices]

    def reduce_tasks(self, small_num_tasks: int):
        assert small_num_tasks <= len(self.train_xs)
        self.train_xs = [x for t_idx, x in enumerate(self.train_xs) if t_idx < small_num_tasks]
        self.val_xs = [x for t_idx, x in enumerate(self.val_xs) if t_idx < small_num_tasks]
        self.test_xs = [x for t_idx, x in enumerate(self.test_xs) if t_idx < small_num_tasks]


if __name__ == '__main__':
    t, s = 10, 1000
    file_name = "../MNIST_coreset/pmc_tasks_{}_size_{}.pkl".format(t, s)
    c = PermutedCoreset(*get_permuted_datasets("PERMUTED_MNIST", t, "../MNIST_data"),
                        sampling_ratio=[(s / 55000), 1.0, 1.0],
                        sampling_type="k-center",
                        load_file_name=file_name)

    if not c.loaded:
        c.dump(file_name)

    cc = deepcopy(c)
    for t in [10, 4, 2]:
        c.reduce_tasks(t)
        for s in [1000, 500, 250, 100]:
            if not (t == 10 and s == 1000):
                c.reduce_xs(s)
                c.dump("../MNIST_coreset/pmc_tasks_{}_size_{}.pkl".format(t, s))
        c = deepcopy(cc)
