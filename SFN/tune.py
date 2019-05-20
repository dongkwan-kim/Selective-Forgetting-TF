from pprint import pprint

from SFN_run import *
from bayes_opt import BayesianOptimization
import numpy as np


def tune_hparam(sfn, n_to_search, _flags):

    utype_to_one_step_units = get_one_step_unit_dict(_flags)
    policy_params = load_params_of_policy(_flags.mtype)

    def black_box_function(tau, mixing_coeff):
        params_of_utype = policy_params.get("OURS")
        params_of_utype["FILTER"]["tau"] = tau
        params_of_utype["NEURON"]["tau"] = tau
        params_of_utype["FILTER"]["mixing_coeff"] = mixing_coeff
        params_of_utype["NEURON"]["mixing_coeff"] = mixing_coeff

        policy_name = "OURS:t{}:mc{}".format(str(round(tau, 5)), str(round(mixing_coeff, 5)))
        try:
            sfn.sequentially_selective_forget_and_predict(
                task_to_forget=_flags.task_to_forget,
                utype_to_one_step_units=utype_to_one_step_units,
                steps_to_forget=_flags.steps_to_forget,
                policy=policy_name,
                params_of_utype=params_of_utype,
                fast_skip=True,
            )
            sfn.recover_old_params()
            return sfn.get_area_under_forgetting_curve(_flags.task_to_forget, policy_name)
        except AssertionError as e:
            mean_rho = float(str(e).split(" = ")[-1])
            if mean_rho > 0.7:
                return - (mean_rho - 0.7)
            else:
                return mean_rho - 0.3

    eps = 1e-7
    param_bounds = {
        "tau": (eps, 0.01),
        "mixing_coeff": (eps, 1 - eps),
    }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=param_bounds,
        random_state=42,
    )
    optimizer.maximize(
        init_points=2,
        n_iter=n_to_search,
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}".format(i))
        pprint(res)
    cprint(optimizer.max, "green")


if __name__ == '__main__':

    params = load_experiment_and_model_params(

        # SFDEN_FORGET, SFDEN_RETRAIN,
        # SFHPS_FORGET, SFHPS_MASK,
        # SFEWC_FORGET,
        # SFLCL10_FORGET, SFLCL10_MASK
        # SFLCL20_FORGET, SFLCL100_FORGET,
        experiment_name="SFDEN_FORGET",

        # SMALL_FC_MNIST,
        # LARGE_FC_MNIST, NOT_XLARGE_FC_MNIST,
        # XLARGE_FC_MNIST
        # SMALL_CONV_MNIST, ALEXNETV_MNIST,
        # ALEXNETV_CIFAR10,
        # ALEXNETV_COARSE_CIFAR100, ALEXNETV_CIFAR100
        model_name="SMALL_FC_MNIST",
    )

    labels, train_xs, val_xs, test_xs, coreset = get_dataset(params.dtype, params)

    model = params.model(params)
    model.add_dataset(labels, train_xs, val_xs, test_xs, coreset)

    if not model.restore():
        model.initial_train()
        if not model.online_importance:
            model.get_importance_matrix(use_coreset=params.need_coreset)
        model.save()

    model.normalize_importance_matrix_about_task()

    # noinspection PyTypeChecker
    tune_hparam(model, 15, params)
