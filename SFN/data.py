import os
from copy import deepcopy
from typing import List, Callable, Tuple
from functools import reduce

import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from termcolor import cprint
from tqdm import trange

from reject.ReusableObject import ReusableObject
from enums import LabelType


class DataLabel:

    def __init__(self, train_labels, validation_labels, test_labels, label_type):
        self.train_labels: np.ndarray or tuple = train_labels
        self.validation_labels: np.ndarray or tuple = validation_labels
        self.test_labels: np.ndarray or tuple = test_labels
        self.label_type: LabelType = label_type

    def get_train_labels(self, task_id) -> np.ndarray:
        return self._get_labels(self.train_labels, task_id)

    def get_validation_labels(self, task_id) -> np.ndarray:
        return self._get_labels(self.validation_labels, task_id)

    def get_test_labels(self, task_id) -> np.ndarray:
        return self._get_labels(self.test_labels, task_id)

    def _get_labels(self, labels, task_id) -> np.ndarray:
        if self.label_type == LabelType.ONE_LABELS_TO_ALL_TASK:
            return labels
        elif self.label_type == LabelType.ONE_LABEL_TO_ONE_CLASS:
            assert task_id is not None and task_id > 0
            return np.asarray([labels[0][task_id - 1] for _ in range(labels[1][task_id - 1])])

    def slice_train_labels(self, indexes: np.ndarray or List[np.ndarray]):
        self.train_labels = self._slice_labels(self.train_labels, indexes)

    def slice_validation_labels(self, indexes: np.ndarray or List[np.ndarray]):
        self.validation_labels = self._slice_labels(self.validation_labels, indexes)

    def slice_test_labels(self, indexes: np.ndarray or List[np.ndarray]):
        self.test_labels = self._slice_labels(self.test_labels, indexes)

    def _slice_labels(self, labels, indexes: np.ndarray or List[np.ndarray]):
        if self.label_type == LabelType.ONE_LABELS_TO_ALL_TASK:
            return labels[indexes]
        elif self.label_type == LabelType.ONE_LABEL_TO_ONE_CLASS:
            real_labels, cls_sz = labels
            sliced_cls_sz = [len(indexes_of_cls) for indexes_of_cls in indexes]
            assert len(cls_sz) == len(sliced_cls_sz)
            return real_labels, sliced_cls_sz


def dtype_to_name(dtype: str):
    return dtype.lower().split("_")[-1]


def name_to_train_and_val_num(name: str) -> tuple:
    if name == "mnist":
        return 55000, 5000
    elif name == "cifar100":
        return 45000, 5000
    else:
        raise ValueError


def get_to_one_hot(num_class: int) -> Callable:
    lb = LabelBinarizer()
    lb.fit(range(num_class))
    return lb.transform


def preprocess_xs(name: str, xs: List[np.ndarray], is_for_cnn=None) -> tuple:

    def reshape_xs():
        _xs = []
        for x in xs:
            _, *whd = x.shape
            # (-1, w * h * d) if not is_for_cnn else (-1, w, h, d)
            x = x.reshape((-1, reduce(lambda i, j: i*j, whd))) if not is_for_cnn else x.reshape((-1, *whd))
            _xs.append(x)
        return _xs

    if name == "mnist":
        is_for_cnn = is_for_cnn or False
        reshaped_xs = reshape_xs()
        return tuple(x / 255 for x in reshaped_xs)
    elif name == "cifar100":
        is_for_cnn = is_for_cnn or True
        reshaped_xs = reshape_xs()
        return tuple(x / 255 for x in reshaped_xs)
    else:
        raise ValueError


def get_tfds(dtype: str, data_dir: str = None, x_name="image", y_name="label", is_verbose=True, **kwargs):

    name = dtype_to_name(dtype)
    assert name in tfds.list_builders()

    data_dir = data_dir or os.path.join("~", "tfds", "{}_data".format(name.upper()))  # e.g. ~/tfds/MNIST_data

    # https://www.tensorflow.org/datasets/datasets
    loaded, info = tfds.load(
        name=name,
        split=["train", "test"],
        data_dir=data_dir,
        batch_size=-1,
        with_info=True,
    )
    if is_verbose:
        print(info)

    # Get numpy matrix
    train_and_validation, test = tfds.as_numpy(loaded)

    # Preprocess & Reshape
    train_and_validation_x, test_x = preprocess_xs(
        name,
        [train_and_validation[x_name], test[x_name]],
        **kwargs,
    )

    # Training Validation Separation
    # this is necessary because tfds does not support validation separation.
    train_num = name_to_train_and_val_num(name)[0]
    train_x = train_and_validation_x[:train_num]
    val_x = train_and_validation_x[train_num:]

    # One hot labeling
    to_one_hot = get_to_one_hot(info.features[y_name].num_classes)
    data_label = DataLabel(
        train_labels=to_one_hot(train_and_validation[y_name][:train_num]),
        validation_labels=to_one_hot(train_and_validation[y_name][train_num:]),
        test_labels=to_one_hot(test[y_name]),
        label_type=LabelType.ONE_LABELS_TO_ALL_TASK,
    )
    return data_label, train_x, val_x, test_x


def get_permuted_datasets(dtype: str, n_tasks: int, data_dir=None, base_seed=42,
                          **kwargs) -> Tuple[DataLabel, list, list, list]:

    data_label, train_x, val_x, test_x = get_tfds(dtype, data_dir, **kwargs)

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


def get_class_as_task_datasets(dtype: str, n_tasks: int, data_dir=None,
                               **kwargs) -> Tuple[DataLabel, list, list, list]:

    data_label, train_x, val_x, test_x = get_tfds(dtype, data_dir, **kwargs)

    train_not_one_hot_label = np.argmax(data_label.train_labels, axis=1)
    validation_not_one_hot_label = np.argmax(data_label.validation_labels, axis=1)
    test_not_one_hot_label = np.argmax(data_label.test_labels, axis=1)

    # Class Dividing
    _train_xs, _val_xs, _test_xs = [], [], []
    for cls_id in range(n_tasks):
        train_indexes_of_cls = train_not_one_hot_label == cls_id
        validation_indexes_of_cls = validation_not_one_hot_label == cls_id
        test_indexes_of_cls = test_not_one_hot_label == cls_id

        _train_xs.append(train_x[train_indexes_of_cls])
        _val_xs.append(val_x[validation_indexes_of_cls])
        _test_xs.append(test_x[test_indexes_of_cls])

    # Construct labels
    to_one_hot = get_to_one_hot(n_tasks)
    data_label = DataLabel(
        train_labels=(to_one_hot(list(range(n_tasks))), [len(x) for x in _train_xs]),
        validation_labels=(to_one_hot(list(range(n_tasks))), [len(x) for x in _val_xs]),
        test_labels=(to_one_hot(list(range(n_tasks))), [len(x) for x in _test_xs]),
        label_type=LabelType.ONE_LABEL_TO_ONE_CLASS
    )
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

    def __init__(self,
                 data_labels: DataLabel,
                 train_xs, val_xs, test_xs,
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

        data_labels.slice_train_labels(train_sampled_idx)
        data_labels.slice_validation_labels(val_sampled_idx)
        data_labels.slice_test_labels(test_sampled_idx)
        self.data_labels = data_labels

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
        return self.train_xs[item], self.get_train_labels(item + 1), \
               self.val_xs[item], self.get_validation_labels(item + 1), \
               self.test_xs[item], self.get_test_labels(item + 1)
    
    def get_train_labels(self, task_id):
        return self.data_labels.get_train_labels(task_id)
    
    def get_validation_labels(self, task_id):
        return self.data_labels.get_validation_labels(task_id)
    
    def get_test_labels(self, task_id):
        return self.data_labels.get_test_labels(task_id)

    def reduce_xs(self, small_sz: int):
        small_indices = np.arange(small_sz)
        self.train_xs = slice_xs(self.train_xs, small_indices)
        self.data_labels.slice_train_labels(small_indices)

    def reduce_tasks(self, small_num_tasks: int):
        assert small_num_tasks <= len(self.train_xs)
        self.train_xs = [x for t_idx, x in enumerate(self.train_xs) if t_idx < small_num_tasks]
        self.val_xs = [x for t_idx, x in enumerate(self.val_xs) if t_idx < small_num_tasks]
        self.test_xs = [x for t_idx, x in enumerate(self.test_xs) if t_idx < small_num_tasks]


if __name__ == '__main__':
    t, s = 10, 1000
    file_name = "~/tfds/MNIST_coreset/pmc_tasks_{}_size_{}.pkl".format(t, s)
    c = PermutedCoreset(*get_permuted_datasets("PERMUTED_MNIST", t, "~/tfds/MNIST_data"),
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
                c.dump("~/tfds/MNIST_coreset/pmc_tasks_{}_size_{}.pkl".format(t, s))
        c = deepcopy(cc)
