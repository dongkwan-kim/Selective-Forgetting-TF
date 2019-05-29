from pprint import pprint

from SFN_run import *
from mask import Mask
from collections import Counter, defaultdict
from utils import build_hist, build_bar


def get_indices(tid, start_idx, batch_size, mask_ids, X, is_training):
    return set(Mask.get_mask_stats_by_idx(model.sess, mask_ids, feed_dict={
        X: model.trainXs[tid][start_idx:start_idx + batch_size],
        is_training: True,
    })["indices"])


def profile_masks(sflcl: SFLCL, file_format=".png"):

    X = tf.get_default_graph().get_tensor_by_name("X:0")
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

    mask_ids = [i for i in range(len(sflcl.conv_dims) // 2 - 1)]

    batch_size_per_task = sflcl.batch_size // sflcl.n_tasks
    batch_num = len(sflcl.trainXs[0]) // batch_size_per_task
    cprint("batch_num: {}".format(batch_num), "red")

    top_indices_of_task_of_mask = defaultdict(list)
    for t in range(sflcl.n_tasks):

        for m in mask_ids:

            start_idx = 0
            all_indices = []
            for _ in range(batch_num):
                indices = get_indices(t, start_idx, batch_size_per_task, m, X, is_training)
                all_indices += list(indices)

                start_idx += batch_size_per_task

            counter = Counter(all_indices)
            counted = list(counter.values())
            build_hist(counted,
                       bins=25, density=True,
                       xlabel="Number of activations", ylabel=None,
                       title="Dist. of # activations",
                       file_name="../figs/activated_filters_{}_layer_{}{}".format(t + 1, m + 1, file_format))

            top_indices = [i for i, c in counter.items() if c > 90]
            cprint("Top indices of task {}, layer {} > {}".format(t + 1, m + 1, 90), "red")
            """
            top_indices = [i for i, c in counter.items() if c > float(np.mean(counted) + np.std(counted))]
            cprint("Top indices of task {}, layer {} > {} = {} + {}".format(
                t + 1, m + 1, np.mean(counted) + np.std(counted), np.mean(counted), np.std(counted)), "red")
            """
            top_indices_of_task_of_mask[m].append(top_indices)

    for m in mask_ids:
        top_indices_flatten = []
        for ti in top_indices_of_task_of_mask[m]:
            top_indices_flatten += ti
        counter = Counter(top_indices_flatten)

        n_task_to_filters = defaultdict(list)
        for i, c in counter.items():
            n_task_to_filters[c].append(i)

        xs = list(range(1, 10 + 1))
        ys = [len(n_task_to_filters[i]) for i in xs]
        dist_ys = [y / sum(ys) for y in ys]
        build_bar(x=xs,
                  y=dist_ys,
                  min_y_pos=1,
                  title="Dist. of # tasks at which FAF activates",
                  file_name="../figs/number_of_tasks_layer{}{}".format(m + 1, file_format),
                  xlabel="Number of tasks",
                  ylabel=None)

    for m, top_indices_of_task in top_indices_of_task_of_mask.items():
        cprint("- Mask {}".format(m), "green")
        for t, indices in enumerate(top_indices_of_task):
            print("{}: total {} / {}".format(t + 1, len(indices), indices))
        szs = [len(indices) for indices in top_indices_of_task]
        print("Mean sz: {} +- {}".format(np.mean(szs), np.std(szs)))


if __name__ == '__main__':

    params = load_experiment_and_model_params(
        experiment_name="SFLCL10_MASK",
        model_name="ALEXNETV_CIFAR10",
    )

    # noinspection PyTypeChecker
    labels, train_xs, val_xs, test_xs, coreset = get_dataset(params.dtype, params)

    model = params.model(params)
    model.add_dataset(labels, train_xs, val_xs, test_xs, coreset)

    if not model.restore():
        model.initial_train()
        if not model.online_importance:
            model.get_importance_matrix(use_coreset=params.need_coreset)
        model.save()

    model.normalize_importance_matrix_about_task()

    profile_masks(model, file_format=".pdf")




