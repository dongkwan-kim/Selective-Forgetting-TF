import os
import tensorflow as tf

from SFN.SFDEN import SFDEN
from SFN.SFHPS import SFHPS
from SFN.SFLCL import SFLCL
from data import *
from utils import build_line_of_list

np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 400, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learning rate(init) for train")
flags.DEFINE_integer("batch_size", 256, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "../checkpoints/default", "Directory path to save the checkpoints")
flags.DEFINE_integer("dims0", 784, "Dimensions about input layer of fully connect layers")
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
flags.DEFINE_integer("task_to_forget", 6, 'Task to forget')
flags.DEFINE_integer("one_step_neurons", 5, 'Number of neurons to forget in one step')
flags.DEFINE_integer("steps_to_forget", 35, 'Total number of steps in forgetting')
flags.DEFINE_string("importance_criteria", "first_Taylor_approximation", "Criteria to measure importance of neurons")
flags.DEFINE_integer("retrain_max_iter_per_task", 150, "Epoch to re-train per one task")
flags.DEFINE_integer("retrain_task_iter", 80, "Number of re-training one task with retrain_max_iter_per_task")
flags.DEFINE_integer("coreset_size", 250, "Size of coreset")

MODE = {
    "SIZE": "DEFAULT",  # TEST, SMALL, DEFAULT
    "EXPERIMENT": "FORGET",  # FORGET, RETRAIN, CRITERIA
    "MODEL": SFLCL,  # SFDEN, SFHPS, SFLCL
    "DTYPE": "MNIST",  # PERMUTED_MNIST, COARSE_CIFAR100, MNIST
}

if MODE["SIZE"] == "TEST":
    flags.FLAGS.max_iter = 90
    flags.FLAGS.n_tasks = 2
    flags.FLAGS.task_to_forget = 1
    flags.FLAGS.steps_to_forget = 6
    flags.FLAGS.checkpoint_dir = "../checkpoints/test"
elif MODE["SIZE"] == "SMALL":
    flags.FLAGS.max_iter = 200
    flags.FLAGS.n_tasks = 4
    flags.FLAGS.task_to_forget = 2
    flags.FLAGS.steps_to_forget = 14
    flags.FLAGS.checkpoint_dir = "../checkpoints/small"

if MODE["EXPERIMENT"] == "RETRAIN":
    flags.FLAGS.steps_to_forget = flags.FLAGS.steps_to_forget - 12
elif MODE["EXPERIMENT"] == "CRITERIA":
    flags.FLAGS.importance_criteria = "activation"
    flags.FLAGS.checkpoint_dir += "/" + flags.FLAGS.importance_criteria

if MODE["DTYPE"] == "COARSE_CIFAR100":
    flags.FLAGS.n_classes = 20
    flags.DEFINE_integer("conv0_filters", 3, "Number of filters in input")
    flags.DEFINE_integer("conv0_size", 32, "Size of input")
    flags.DEFINE_integer("conv1_filters", 64, "Number of filters in conv1")
    flags.DEFINE_integer("conv1_size", 5, "Size of kernel in conv1")
    flags.DEFINE_integer("pool1_ksize", 2, "Size of pooling window for xy direction of images")
    flags.DEFINE_integer("conv2_filters", 64, "Number of filters in conv2")
    flags.DEFINE_integer("conv2_size", 5, "Size of kernel in conv2")
    flags.DEFINE_integer("pool2_ksize", 2, "Size of pooling window for xy direction of images")
    flags.DEFINE_integer("conv3_filters", 128, "Number of filters in conv3")
    flags.DEFINE_integer("conv3_size", 3, "Size of kernel in conv3")
    flags.DEFINE_integer("conv4_filters", 128, "Number of filters in conv4")
    flags.DEFINE_integer("conv4_size", 3, "Size of kernel in conv4")
    flags.DEFINE_integer("conv5_filters", 128, "Number of filters in conv5")
    flags.DEFINE_integer("conv5_size", 3, "Size of kernel in conv5")
    flags.DEFINE_integer("fc0", 8*8*128, "Dimensions about input layer of fully connect layers")
    flags.DEFINE_integer("fc1", 128, "Dimensions about 1st layer")
    flags.DEFINE_integer("fc2", flags.FLAGS.n_classes, "Dimensions of output layer")
elif MODE["DTYPE"] == "MNIST":
    flags.DEFINE_integer("conv0_filters", 1, "Number of filters in input")
    flags.DEFINE_integer("conv0_size", 28, "Size of input")
    flags.DEFINE_integer("conv1_filters", 3, "Number of filters in conv1")
    flags.DEFINE_integer("conv1_size", 5, "Size of kernel in conv1")
    flags.DEFINE_integer("pool1_ksize", 2, "Size of pooling window for xy direction of images")
    flags.DEFINE_integer("fc0", 14*14*3, "Dimensions about input layer of fully connect layers")
    flags.DEFINE_integer("fc1", 128, "Dimensions about 1st layer")
    flags.DEFINE_integer("fc2", flags.FLAGS.n_classes, "Dimensions of output layer")

flags.FLAGS.checkpoint_dir = os.path.join(flags.FLAGS.checkpoint_dir, MODE["MODEL"].__name__)
if MODE["MODEL"] == SFHPS:
    flags.FLAGS.max_iter = 1
    flags.FLAGS.dims1 += 10 * flags.FLAGS.n_tasks
    flags.FLAGS.dims2 += 10 * flags.FLAGS.n_tasks
    flags.FLAGS.retrain_task_iter = 1000
    flags.FLAGS.one_step_neurons = 7
    flags.FLAGS.l1_lambda = 0.00001
    flags.FLAGS.l2_lambda = 0.0
elif MODE["MODEL"] == SFLCL:
    flags.FLAGS.max_iter = 800
    flags.FLAGS.n_tasks = flags.FLAGS.n_classes


FLAGS = flags.FLAGS


def experiment_forget(sfn, _flags, _policies):
    for policy in _policies:
        sfn.sequentially_selective_forget_and_predict(
            _flags.task_to_forget, _flags.one_step_neurons, _flags.steps_to_forget,
            policy=policy,
        )
        sfn.recover_old_params()

    sfn.print_summary(_flags.task_to_forget, _flags.one_step_neurons)
    sfn.draw_chart_summary(_flags.task_to_forget, _flags.one_step_neurons,
                           file_prefix="../figs/{}_task{}_step{}_total{}".format(
                               sfn.__class__.__name__,
                               _flags.task_to_forget,
                               _flags.one_step_neurons,
                               str(int(_flags.steps_to_forget) * int(_flags.one_step_neurons)),
                           ))


def experiment_forget_and_retrain(sfn, _flags, _policies, _coreset=None):
    one_shot_neurons = _flags.one_step_neurons * _flags.steps_to_forget
    for policy in _policies:
        sfn.sequentially_selective_forget_and_predict(
            _flags.task_to_forget, one_shot_neurons, 1,
            policy=policy,
        )
        lst_of_perfs_at_epoch = sfn.retrain_after_forgetting(
            _flags, policy, _coreset,
            epoches_to_print=[0, 1, -2, -1],
            is_verbose=False,
        )
        build_line_of_list(x=list(i * _flags.retrain_max_iter_per_task for i in range(len(lst_of_perfs_at_epoch))),
                           y_list=np.transpose(lst_of_perfs_at_epoch),
                           label_y_list=[t + 1 for t in range(_flags.n_tasks)],
                           xlabel="Re-training Epoches", ylabel="AUROC", ylim=[0.9, 1],
                           title="AUROC By Retraining After Forgetting Task-{} ({})".format(
                               _flags.task_to_forget,
                               policy,
                           ),
                           file_name="../figs/{}_{}_task{}_RetrainAcc.png".format(
                               sfn.__class__.__name__,
                               sfn.importance_criteria.split("_")[0],
                               _flags.task_to_forget,
                           ))
        sfn.clear_experiments()


def get_dataset(dtype: str, _flags, **kwargs) -> tuple:

    if dtype == "PERMUTED_MNIST":
        _labels, _train_xs, _val_xs, _test_xs = get_permuted_datasets(dtype, _flags.n_tasks, **kwargs)
        train_sz = _train_xs[0].shape[0]
        _coreset = PermutedCoreset(
            _labels, _train_xs, _val_xs, _test_xs,
            sampling_ratio=[(_flags.coreset_size / train_sz), 1.0, 1.0],
            sampling_type="k-center",
            load_file_name=os.path.join("~/tfds/MNIST_coreset", "pmc_tasks_{}_size_{}.pkl".format(
                _flags.n_tasks,
                _flags.coreset_size,
            )),
        )
        return _labels, _train_xs, _val_xs, _test_xs, _coreset

    elif dtype == "COARSE_CIFAR100":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks,
                                                                           y_name="coarse_label", **kwargs)
        _coreset = None  # TODO
        return _labels, _train_xs, _val_xs, _test_xs, _coreset

    elif dtype == "MNIST":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks,
                                                                           is_for_cnn=True, **kwargs)
        _coreset = None  # TODO
        return _labels, _train_xs, _val_xs, _test_xs, _coreset

    else:
        raise ValueError


if __name__ == '__main__':

    labels, train_xs, val_xs, test_xs, coreset = get_dataset(MODE["DTYPE"], FLAGS)

    model = MODE["MODEL"](FLAGS)
    model.add_dataset(labels, train_xs, val_xs, test_xs)

    if not model.restore():
        model.initial_train()
        model.get_importance_matrix()
        model.save()

    if MODE["EXPERIMENT"] == "FORGET" or MODE["EXPERIMENT"] == "CRITERIA":
        policies_for_expr = ["MIX", "MAX", "VAR", "LIN", "EIN", "RANDOM", "ALL", "ALL_VAR"]
        experiment_forget(model, FLAGS, policies_for_expr)
    elif MODE["EXPERIMENT"] == "RETRAIN":
        policies_for_expr = ["MIX"]
        experiment_forget_and_retrain(model, FLAGS, policies_for_expr, coreset)
