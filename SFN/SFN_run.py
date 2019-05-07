from typing import Dict

import os
import tensorflow as tf

from SFDEN import SFDEN
from SFHPS import SFHPS
from SFLCL import SFLCL
from params import MyParams, check_params, to_yaml_path
from data import *
from utils import build_line_of_list, get_project_dir
from enums import UnitType

np.random.seed(1004)


def load_experiment_and_model_params() -> MyParams:
    loaded_params = MyParams(
        yaml_file_to_config_name={

            # SFDEN_FORGET, SFDEN_RETRAIN, SFHPS_FORGET,
            # SFLCL10_FORGET, SFLCL20_FORGET, SFLCL100_FORGET
            to_yaml_path("experiment.yaml"): "SFHPS_FORGET",

            # SMALL_FC_MNIST, LARGE_FC_MNIST,
            # SMALL_CONV_MNIST, ALEXNETV_MNIST,
            # ALEXNETV_CIFAR10, ALEXNETV_COARSE_CIFAR100, ALEXNETV_CIFAR100
            to_yaml_path("models.yaml"): "LARGE_FC_MNIST",

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


def load_params_of_policy(mtype, **kwargs) -> MyParams:
    loaded_params = MyParams(
        yaml_file_to_config_name={to_yaml_path("policies.yaml"): mtype},
        **kwargs
    )
    return loaded_params


def get_one_step_unit_dict(_flags, is_one_shot=False) -> Dict[UnitType, int]:
    utype_to_step_dict = {}
    for utype in UnitType:
        one_step_key = "one_step_{}s".format(str(utype).lower())
        if _flags.has(one_step_key):
            if not is_one_shot:
                utype_to_step_dict[utype] = _flags.get(one_step_key)
            else:
                utype_to_step_dict[utype] = _flags.get(one_step_key) * _flags.steps_to_forget

    assert len(utype_to_step_dict) != 0
    return utype_to_step_dict


def experiment_forget(sfn, _flags, _policies):

    utype_to_one_step_units = get_one_step_unit_dict(_flags)
    policy_params = load_params_of_policy(_flags.mtype)

    for policy in _policies:
        sfn.sequentially_selective_forget_and_predict(
            task_to_forget=_flags.task_to_forget,
            utype_to_one_step_units=utype_to_one_step_units,
            steps_to_forget=_flags.steps_to_forget,
            policy=policy,
            params_of_utype=policy_params.get(policy),
        )
        sfn.recover_old_params()

    sfn.print_summary(_flags.task_to_forget)
    sfn.draw_chart_summary(
        _flags.task_to_forget,
        file_prefix=os.path.join(get_project_dir(), "figs/{}_{}_task{}".format(
            _flags.model.__name__, _flags.expr_type, _flags.task_to_forget
        )),
        file_extension=".pdf",
        highlight_ylabels=["OURS"],
    )


def experiment_forget_and_retrain(sfn, _flags, _policies, _coreset=None):
    policy_params = load_params_of_policy(_flags.mtype)
    for policy in _policies:
        sfn.sequentially_selective_forget_and_predict(
            task_to_forget=_flags.task_to_forget,
            utype_to_one_step_units=get_one_step_unit_dict(_flags, is_one_shot=True),
            steps_to_forget=1,
            policy=policy,
            params_of_utype=policy_params.get(policy),
        )
        lst_of_perfs_at_epoch = sfn.retrain_after_forgetting(
            _flags, policy, _coreset,
            epoches_to_print=[0, 1, -2, -1],
            is_verbose=False,
        )
        build_line_of_list(x_or_x_list=list(i * _flags.retrain_max_iter_per_task
                                            for i in range(len(lst_of_perfs_at_epoch))),
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
    # Note that coreset is necessary for the continual learning
    if dtype == "PERMUTED_MNIST":
        _labels, _train_xs, _val_xs, _test_xs = get_permuted_datasets(dtype, _flags.n_tasks, **kwargs)
        train_sz = _train_xs[0].shape[0]
        if _flags.need_coreset:
            _coreset = Coreset(
                _labels, _train_xs, _val_xs, _test_xs,
                sampling_ratio=[(_flags.coreset_size / train_sz), 1.0, 1.0],
                sampling_type="k-center",
                load_file_name=os.path.join("~/tfds/PERMUTED_MNIST_coreset",
                                            "coreset_size_{}.pkl".format(_flags.coreset_size)),
            )
        else:
            _coreset = None

    elif dtype == "COARSE_CIFAR100":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks,
                                                                           y_name="coarse_label", **kwargs)
        _coreset = None

    elif dtype == "CIFAR10":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks, **kwargs)
        _coreset = None

    elif dtype == "CIFAR100":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks, **kwargs)
        _coreset = None

    elif dtype == "MNIST":
        _labels, _train_xs, _val_xs, _test_xs = get_class_as_task_datasets(dtype, _flags.n_tasks,
                                                                           is_for_cnn=True, **kwargs)
        _coreset = None
    else:
        raise ValueError

    return _labels, _train_xs, _val_xs, _test_xs, _coreset


if __name__ == '__main__':

    params = load_experiment_and_model_params()

    # noinspection PyTypeChecker
    labels, train_xs, val_xs, test_xs, coreset = get_dataset(params.dtype, params)

    model = params.model(params)
    model.add_dataset(labels, train_xs, val_xs, test_xs)

    if not model.restore():
        model.initial_train()
        model.get_importance_matrix()
        model.save()

    model.normalize_importance_matrix_about_task()

    if params.expr_type == "FORGET" or params.expr_type == "CRITERIA":
        policies_for_expr = ["RANDOM", "MEAN", "MAX", "CONST", "OURS"]
        # noinspection PyTypeChecker
        experiment_forget(model, params, policies_for_expr)
    elif params.expr_type == "RETRAIN":
        policies_for_expr = ["OURS"]
        # noinspection PyTypeChecker
        experiment_forget_and_retrain(model, params, policies_for_expr, coreset)
