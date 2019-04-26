import os
import tensorflow as tf

from SFDEN import SFDEN
from SFHPS import SFHPS
from SFLCL import SFLCL
from params import MyParams, check_params, to_yaml_path
from data import *
from utils import build_line_of_list, get_project_dir

np.random.seed(1004)


def get_load_params() -> MyParams:
    loaded_params = MyParams(
        yaml_file_to_config_name={

            # SFDEN_FORGET, SFDEN_RETRAIN, SFHPS_FORGET,
            # SFLCL10_FORGET, SFLCL20_FORGET
            to_yaml_path("experiment.yaml"): "SFLCL20_FORGET",

            # SMALL_FC_MNIST, LARGE_FC_MNIST,
            # SMALL_CONV_MNIST, ALEXNETV_MNIST,
            # ALEXNETV_CIFAR10, ALEXNETV_COARSE_CIFAR100
            to_yaml_path("models.yaml"): "ALEXNETV_COARSE_CIFAR100",

        },
        value_magician={
            "model": lambda p: {
                "SFDEN": SFDEN,
                "SFHPS": SFHPS,
                "SFLCL": SFLCL,
            }[p.model],
            "checkpoint_dir": lambda p: os.path.join(
                get_project_dir(), p.checkpoint_dir, p.model, p.mtype,
            ),
        })
    check_params(loaded_params)
    loaded_params.pprint()
    return loaded_params


def experiment_forget(sfn, _flags, _policies):
    for policy in _policies:
        sfn.sequentially_selective_forget_and_predict(
            _flags.task_to_forget, _flags.one_step_neurons, _flags.steps_to_forget,
            policy=policy,
        )
        sfn.recover_old_params()

    sfn.print_summary(_flags.task_to_forget, _flags.one_step_neurons)
    sfn.draw_chart_summary(_flags.task_to_forget, _flags.one_step_neurons,
                           file_prefix=os.path.join(
                               get_project_dir(),
                               "figs/{}_task{}_step{}_total{}".format(
                                   sfn.__class__.__name__,
                                   _flags.task_to_forget,
                                   _flags.one_step_neurons,
                                   str(int(_flags.steps_to_forget) * int(_flags.one_step_neurons)),
                               ),
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
                           file_name=os.path.join(
                               get_project_dir(),
                               "figs/{}_{}_task{}_RetrainAcc.png".format(
                                   sfn.__class__.__name__,
                                   sfn.importance_criteria.split("_")[0],
                                   _flags.task_to_forget,
                               ),
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

    elif dtype == "CIFAR10":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks, **kwargs)
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

    params = get_load_params()

    labels, train_xs, val_xs, test_xs, coreset = get_dataset(params.dtype, params)

    model = params.model(params)
    model.add_dataset(labels, train_xs, val_xs, test_xs)

    if not model.restore():
        model.initial_train()
        model.get_importance_matrix()
        model.save()

    if params.expr_type == "FORGET" or params.expr_type == "CRITERIA":
        policies_for_expr = ["MIX", "MAX", "VAR", "LIN", "EIN", "RANDOM", "ALL", "ALL_VAR"]
        # noinspection PyTypeChecker
        experiment_forget(model, params, policies_for_expr)
    elif params.expr_type == "RETRAIN":
        policies_for_expr = ["MIX"]
        # noinspection PyTypeChecker
        experiment_forget_and_retrain(model, params, policies_for_expr, coreset)
