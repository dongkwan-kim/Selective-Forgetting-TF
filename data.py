from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def get_data(n_tasks: int, mnist_dir: str = "./MNIST_data", base_seed=42):

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
