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
        self.temp_coreset_x = []
        self.temp_coreset_y = []
        self.train_xs = train_xs
        self.temp_train_xs = [None]*len(train_xs)
        self.val_xs = val_xs
        self.test_xs = test_xs
        self.y_train = mnist.train.labels
        self.train_labels = []
        self.val_labels = mnist.validation.labels
        self.test_labels = mnist.test.labels
        
        train_sz, val_sz, test_sz = tuple(map(lambda l: len(l[0]), [train_xs, val_xs, test_xs]))
        if isinstance(sampling_ratio, float):
            self.sampling_ratio_list = [sampling_ratio] * 3
        elif isinstance(sampling_ratio, list):
            self.sampling_ratio_list = sampling_ratio
        else:
            raise TypeError


        np.random.seed(seed)
        self.num_tasks = len(self.train_xs)
        for task in range(len(train_xs)):

            self.temp_train_xs[task], temp_train_label = self.k_center(self.train_xs[task], self.y_train, 100)
            self.train_labels.append(temp_train_label)
        print(np.shape(self.temp_train_xs))
        print(np.shape(self.train_labels))
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

    def update_distance(self, dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists
    
    def k_center(self, x_train, y_train, coreset_size):
        # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        x_coreset = []
        y_coreset = []
        dists = np.full(x_train.shape[0], np.inf)
        current_id = 0
        dists = self.update_distance(dists, x_train, current_id)
        idx = [ current_id ]

        for i in range(1, coreset_size):
            current_id = np.argmax(dists)
            dists = self.update_distance(dists, x_train, current_id)
            idx.append(current_id)

        x_coreset.append(x_train[idx,:])
        y_coreset.append(y_train[idx,:])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)

        return x_coreset[0], y_coreset[0]

    def __getitem__(self, item):
        return self.temp_train_xs[item], self.train_labels, \
               self.val_xs[item], self.val_labels, \
               self.test_xs[item], self.test_labels

    @staticmethod
    def slice_xs(xs, indices):
        return list(map(lambda ndarr: ndarr[indices], xs))

    """ K-center coreset selection """

if __name__ == '__main__':
    c = MNISTCoreset(*get_data_of_multiple_tasks(3, "../MNIST_data"), sampling_ratio=0.1)
