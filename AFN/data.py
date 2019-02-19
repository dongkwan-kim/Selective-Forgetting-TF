from typing import List

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from termcolor import cprint


def get_data_of_multiple_tasks(n_tasks: int, mnist_dir: str = "./MNIST_data", base_seed=42) -> tuple:

    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)
    train_x = mnist.train.images
    val_x = mnist.validation.images
    test_x = mnist.test.images

    task_permutation = []
    for task in range(n_tasks):
        np.random.seed(task + base_seed)
        task_permutation.append(np.random.permutation(784))

    _train_xs, _val_xs, _test_xs = [], [], []
    for task in range(n_tasks):
        _train_xs.append(train_x[:, task_permutation[task]])
        _val_xs.append(val_x[:, task_permutation[task]])
        _test_xs.append(test_x[:, task_permutation[task]])

    return mnist, _train_xs, _val_xs, _test_xs


def sample_indices(sz, ratio):
    return np.random.choice(np.arange(sz, dtype=int), size=int(sz * ratio), replace=False)


class MNISTCoreset:

    def __init__(self, mnist, train_xs, val_xs, test_xs,
                 sampling_ratio: float or List[float],
                 sampling_type: str = None,
                 seed: int = 42):

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
        else:
            raise ValueError

        self.train_xs = self.slice_xs(train_xs, train_sampled_idx)
        self.val_xs = self.slice_xs(val_xs, val_sampled_idx)
        self.test_xs = self.slice_xs(test_xs, test_sampled_idx)

        self.train_labels = mnist.train.labels[train_sampled_idx]
        self.val_labels = mnist.validation.labels[val_sampled_idx]
        self.test_labels = mnist.test.labels[test_sampled_idx]

        self.num_tasks = len(self.train_xs)

        cprint("""{name} initialized\
                  \n - num_tasks: {num_tasks}\
                  \n - train: {train_len} ({train_ratio})\
                  \n - validation: {val_len} ({val_ratio})\
                  \n - test: {test_len} ({test_ratio}) \n""".format(**{
            "name": self.__class__.__name__,
            "num_tasks": self.num_tasks,
            "train_len": train_sz,
            "val_len": val_sz,
            "test_len": test_sz,
            "train_ratio": self.sampling_ratio_list[0],
            "val_ratio": self.sampling_ratio_list[1],
            "test_ratio": self.sampling_ratio_list[2],
        }), "blue")

    @staticmethod
    def slice_xs(xs, indices):
        return list(map(lambda ndarr: ndarr[indices], xs))

    def __getitem__(self, item):
        return self.train_xs[item], self.train_labels, \
               self.val_xs[item], self.val_labels, \
               self.test_xs[item], self.test_labels



if __name__ == '__main__':
    c = MNISTCoreset(*get_data_of_multiple_tasks(3, "../MNIST_data"), sampling_ratio=0.1)
