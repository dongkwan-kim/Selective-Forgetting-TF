from pprint import pprint

from SFN_run import *
from bayes_opt import BayesianOptimization
import numpy as np


def tune_policy_param(sfn, n_to_search, _flags):

    utype_to_one_step_units = get_one_step_unit_dict(_flags)
    policy_params = load_params_of_policy(_flags.mtype)

    def black_box_function(taux100, mixing_coeff):

        tau = float(taux100 / 100)

        params_of_utype = policy_params.get("OURS")
        params_of_utype["FILTER"]["tau"] = tau
        params_of_utype["NEURON"]["tau"] = tau
        params_of_utype["FILTER"]["mixing_coeff"] = mixing_coeff
        params_of_utype["NEURON"]["mixing_coeff"] = mixing_coeff

        policy_name = "OURS:t{}:mc{}".format(str(round(tau, 7)), str(round(mixing_coeff, 7)))
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
            au_mean_fc, au_min_fc = sfn.get_area_under_forgetting_curve(_flags.task_to_forget, policy_name)
            return 0.2 * au_mean_fc + 0.8 * au_min_fc
        except AssertionError as e:
            mean_rho = float(str(e).split(" = ")[-1])
            if mean_rho > 0.9:
                return - (mean_rho - 0.9)
            else:
                return mean_rho - 0.1

    eps = 5e-6
    param_bounds = {
        "taux100": (eps, 1 - eps),
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
        res["params"]["tau"] = float(res["params"]["taux100"] / 100)
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
    tune_policy_param(model, 20, params)