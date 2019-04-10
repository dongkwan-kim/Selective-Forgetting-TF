import tensorflow as tf

from SFN.SFDEN import SFDEN
from data import *
from utils import build_line_of_list

np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 400, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learning rate(init) for train")
flags.DEFINE_integer("batch_size", 256, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "../checkpoints/default", "Directory path to save the checkpoints")
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
flags.DEFINE_integer("task_to_forget", 6, 'Task to forget')
flags.DEFINE_integer("one_step_neurons", 5, 'Number of neurons to forget in one step')
flags.DEFINE_integer("steps_to_forget", 35, 'Total number of steps in forgetting')
flags.DEFINE_string("importance_criteria", "first_Taylor_approximation", "Criteria to measure importance of neurons")
flags.DEFINE_integer("retrain_max_iter_per_task", 150, "Epoch to re-train per one task")
flags.DEFINE_integer("retrain_task_iter", 80, "Number of re-training one task with retrain_max_iter_per_task")

MODE = "DEFAULT_FORGET"

if MODE.startswith("TEST"):
    flags.FLAGS.max_iter = 90
    flags.FLAGS.n_tasks = 2
    flags.FLAGS.task_to_forget = 1
    flags.FLAGS.steps_to_forget = 6
    flags.FLAGS.checkpoint_dir = "../checkpoints/test"
elif MODE.startswith("SMALL"):
    flags.FLAGS.max_iter = 200
    flags.FLAGS.n_tasks = 4
    flags.FLAGS.task_to_forget = 2
    flags.FLAGS.steps_to_forget = 14
    flags.FLAGS.checkpoint_dir = "../checkpoints/small"

if MODE.endswith("RETRAIN"):
    flags.FLAGS.steps_to_forget = flags.FLAGS.steps_to_forget - 12
elif MODE.endswith("CRITERIA"):
    flags.FLAGS.importance_criteria = "activation"
    flags.FLAGS.checkpoint_dir += "/" + flags.FLAGS.importance_criteria

FLAGS = flags.FLAGS


def experiment_forget(sfn, flags, policies):
    for policy in policies:
        sfn.sequentially_selective_forget_and_predict(
            flags.task_to_forget, flags.one_step_neurons, flags.steps_to_forget,
            policy=policy,
        )
        sfn.recover_old_params()

    sfn.print_summary(flags.task_to_forget, flags.one_step_neurons)
    sfn.draw_chart_summary(flags.task_to_forget, flags.one_step_neurons,
                           file_prefix="../figs/task{}_step{}_total{}".format(
                               flags.task_to_forget,
                               flags.one_step_neurons,
                               str(int(flags.steps_to_forget) * int(flags.one_step_neurons)),
                           ))


def experiment_forget_and_retrain(sfn, flags, policies, coreset=None):
    one_shot_neurons = flags.one_step_neurons * flags.steps_to_forget
    for policy in policies:
        sfn.sequentially_selective_forget_and_predict(
            flags.task_to_forget, one_shot_neurons, 1,
            policy=policy,
        )
        lst_of_perfs_at_epoch = sfn.retrain_after_forgetting(
            flags, policy, coreset,
            epoches_to_print=[0, 1, -2, -1],
            is_verbose=False,
        )
        build_line_of_list(x=list(i * flags.retrain_max_iter_per_task for i in range(len(lst_of_perfs_at_epoch))),
                           y_list=np.transpose(lst_of_perfs_at_epoch),
                           label_y_list=[t + 1 for t in range(flags.n_tasks)],
                           xlabel="Re-training Epoches", ylabel="Accuracy", ylim=[0.9, 1],
                           title="Accuracy By Retraining After Forgetting Task-{} ({})".format(
                               flags.task_to_forget,
                               policy,
                           ),
                           file_name="../figs/{}_task{}_RetrainAcc.png".format(
                               sfn.importance_criteria.split("_")[0],
                               flags.task_to_forget,
                           ))
        sfn.clear_experiments()


def experiment_bayesian_optimization(sfn, flags, policies, coreset=None, retraining=True, **kwargs):
    for policy in policies:
        optimal_number_of_neurons = sfn.optimize_number_of_neurons(
            flags.task_to_forget, policy, **kwargs,
        )
        sfn.sequentially_selective_forget_and_predict(
            flags.task_to_forget, optimal_number_of_neurons, 1,
            policy=policy,
        )
        if retraining:
            sfn.retrain_after_forgetting(flags, policy, coreset)
        sfn.clear_experiments()


if __name__ == '__main__':

    mnist_data, train_xs, val_xs, test_xs = get_permuted_mnist_datasets(FLAGS.n_tasks)
    mnist_coreset = PermutedMNISTCoreset(
        mnist_data, train_xs, val_xs, test_xs,
        sampling_ratio=[(250 / 55000), 1.0, 1.0],
        sampling_type="k-center",
        load_file_name="../MNIST_coreset/pmc_tasks_{}_size_250.pkl".format(FLAGS.n_tasks),
    )

    model = SFDEN(FLAGS)
    model.add_dataset(mnist_data, train_xs, val_xs, test_xs)

    if not model.restore():
        model.train_den(FLAGS)
        model.get_importance_matrix()
        model.save()

    if MODE.endswith("FORGET") or MODE.endswith("CRITERIA"):
        policies_for_expr = ["MIX", "MAX", "VAR", "LIN", "EIN", "RANDOM", "ALL", "ALL_VAR"]
        experiment_forget(model, FLAGS, policies_for_expr)
    elif MODE.endswith("RETRAIN"):
        policies_for_expr = ["MIX"]
        experiment_forget_and_retrain(model, FLAGS, policies_for_expr, mnist_coreset)
