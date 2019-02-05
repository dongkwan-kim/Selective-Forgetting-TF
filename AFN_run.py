import tensorflow as tf
import numpy as np
import AFN
from tensorflow.examples.tutorials.mnist import input_data
from pprint import pprint
from MatplotlibUtil import *

np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 400, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
flags.DEFINE_integer("batch_size", 256, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory path to save the checkpoints")
flags.DEFINE_integer("dims0", 784, "Dimensions about input layer")
flags.DEFINE_integer("dims1", 64, "Dimensions about 1st layer")
flags.DEFINE_integer("dims2", 32, "Dimensions about 2nd layer")
flags.DEFINE_integer("dims3", 10, "Dimensions about output layer")
flags.DEFINE_integer("n_classes", 10, 'The number of classes at each task')
flags.DEFINE_float("l1_lambda", 0.00001, "Sparsity for L1")
flags.DEFINE_float("l2_lambda", 0.0001, "L2 lambda")
flags.DEFINE_float("gl_lambda", 0.001, "Group Lasso lambda")
flags.DEFINE_float("regular_lambda", 0.5, "regularization lambda")
flags.DEFINE_integer("ex_k", 10, "The number of units increased in the expansion processing")
flags.DEFINE_float('loss_thr', 0.1, "Threshold of dynamic expansion")
flags.DEFINE_float('spl_thr', 0.1, "Threshold of split and duplication")

# New hyper-parameters
flags.DEFINE_integer("n_tasks", 10, 'The number of tasks')
FLAGS = flags.FLAGS


def get_data(n_tasks: int, mnist_dir: str = "./MNIST_data"):
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)
    train_x = mnist.train.images
    val_x = mnist.validation.images
    test_x = mnist.test.images

    task_permutation = []
    for task in range(n_tasks):
        task_permutation.append(np.random.permutation(784))

    _train_xs, _val_xs, _test_xs = [], [], []
    for task in range(n_tasks):
        _train_xs.append(train_x[:, task_permutation[task]])
        _val_xs.append(val_x[:, task_permutation[task]])
        _test_xs.append(test_x[:, task_permutation[task]])

    return mnist, _train_xs, _val_xs, _test_xs


if __name__ == '__main__':

    mnist_data, train_xs, val_xs, test_xs = get_data(FLAGS.n_tasks)
    model = AFN.AFN(FLAGS)
    model.add_dataset(mnist_data, train_xs, val_xs, test_xs)
    model.train_den(FLAGS)
    model.get_importance_matrix()

    print(model.predict_only_after_training())

    task_to_forget = 6
    one_step_neurons = 5
    steps = 25

    model.sequentially_adaptive_forget_and_predict(task_to_forget, one_step_neurons, steps, policy="RANDOM")
    model.recover_old_params()
    model.sequentially_adaptive_forget_and_predict(task_to_forget, one_step_neurons, steps, policy="LIN")
    model.recover_old_params()
    model.sequentially_adaptive_forget_and_predict(task_to_forget, one_step_neurons, steps, policy="EIN")
    model.recover_old_params()
    model.sequentially_adaptive_forget_and_predict(task_to_forget, one_step_neurons, steps, policy="ALL")
    model.recover_old_params()
    model.print_summary(task_to_forget, one_step_neurons)
    model.draw_chart_summary(task_to_forget, one_step_neurons,
                             file_prefix="task{}_step{}".format(task_to_forget, one_step_neurons))
