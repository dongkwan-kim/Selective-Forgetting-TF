from typing import List, Dict, Tuple

from copy import deepcopy
from termcolor import cprint
from pprint import pprint
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import re

from mask import Mask


def get_mean_and_std_wo_indices(ndarr: list or np.ndarray, indices=None) -> Tuple[float, float]:
    if indices is not None:
        ndarr = np.delete(ndarr, indices)
    return float(np.mean(ndarr)), float(np.std(ndarr))


def sum_set(s: set, *args):
    s = deepcopy(s)
    for _s in args:
        s.update(_s)
    return s


def get_gpu_utility(gpu_id):
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")

    seen_id = -1
    for item in out_list:
        items = [x.strip() for x in item.split(':')]
        if len(items) == 2:
            key, val = items
            if key == "Minor Number":
                seen_id = int(val)
            if seen_id == gpu_id and key == "Gpu":
                return int(val.split(" ")[0])
    else:
        raise EnvironmentError("There's no GPU of no {}".format(gpu_id))


def get_batch_iterator(xs_or_ys: np.ndarray, bsz: int):
    c = cycle(xs_or_ys)

    def batch_iterator():
        return np.asarray([next(c) for _ in range(bsz)])

    return batch_iterator


# Project Utils

def get_project_dir():
    return os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))


def get_dims_from_config(config, search="dims", with_key=False) -> list:
    dims_config = sorted([(k, v) for k, v in config.values().items() if search in k],
                         key=lambda t: t[0])
    if not with_key:
        return [v for _, v in dims_config]
    else:
        return dims_config


def get_middle_path_name(name_to_bool_variable: Dict[str, bool], adapter_str: str = "-") -> str:
    path_elements = []
    for k, v in sorted(name_to_bool_variable.items()):
        if v:
            path_elements.append(k)
    return adapter_str.join(str(x) for x in path_elements)


def cprint_stats_of_mask_pair(model, task_id_a, task_id_b,
                              batch_size_per_task, X, is_training, mask_id=1):
    def get_indices(tid, si):
        return set(Mask.get_mask_stats_by_idx(model.sess, mask_id, feed_dict={
            X: model.trainXs[tid][si:si + batch_size_per_task],
            is_training: True,
        })["indices"])

    gi_a_1 = get_indices(task_id_a - 1, 0)
    gi_a_2 = get_indices(task_id_a - 1, batch_size_per_task)
    gi_a_3 = get_indices(task_id_a - 1, 2 * batch_size_per_task)
    int_a = gi_a_1 & gi_a_2 & gi_a_3
    uni_a = gi_a_1 | gi_a_2 | gi_a_3

    gi_b_1 = get_indices(task_id_b - 1, 0)
    gi_b_2 = get_indices(task_id_b - 1, batch_size_per_task)
    gi_b_3 = get_indices(task_id_b - 1, 2 * batch_size_per_task)
    int_b = gi_b_1 & gi_b_2 & gi_b_3
    uni_b = gi_b_1 | gi_b_2 | gi_b_3

    cprint("Mask{} for tasks {} and {}".format(mask_id, task_id_a, task_id_b), "yellow")
    cprint("Task{} size: {} / {} / {}".format(task_id_a, len(gi_a_1), len(gi_a_2), len(gi_a_3)), "red")
    cprint("Task{} size: {} / {} / {}".format(task_id_b, len(gi_b_1), len(gi_b_2), len(gi_b_3)), "blue")
    cprint("Task{}: intersection {} / union {}".format(task_id_a, len(int_a), len(uni_a)), "red")
    cprint("Task{}: intersection {} / union {}".format(task_id_b, len(int_b), len(uni_b)), "blue")
    cprint("Symmetric Diff {}<->{}: intersect {} over {} / union {} over {}".format(
        task_id_a, task_id_b,
        len(int_a ^ int_b), len(int_a | int_b), len(uni_a ^ uni_b), len(uni_a | uni_b)
    ), "yellow")
    cprint("--------------------------------------", "yellow")


# Tensorflow Utils

def parse_var_name(var_name):
    # prefix, task_id, scope, var_type
    p = re.compile("(.+)_t(\\d+)_(\\w+)/(.+):0")
    return [x if not x.isnumeric() else int(x) for x in p.findall(var_name)[0]]


def print_all_vars(prefix: str = None, color=None):
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    if prefix:
        cprint(prefix, color)
    pprint(all_variables)


def print_ckpt_vars(model_path, prefix: str = None, color=None):
    vars_in_checkpoint = tf.train.list_variables(model_path)
    if prefix:
        cprint(prefix, color)
    pprint(vars_in_checkpoint)


def get_available_gpu_names(gpu_num_list: List[int] = None) -> List[str]:
    """
    :param gpu_num_list e.g. [1, 2]
    :return e.g. ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']

    Ref. https://stackoverflow.com/a/38580201
    """
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    # TODO: Not use right now.
    # if gpu_num_list is not None:
    #    gpu_names = [x for x in gpu_names if int(x.split(":")[-1]) in gpu_num_list]

    return gpu_names


def with_tf_device_gpu(func):
    def wrapped(*args, **kwargs):
        x = None
        model = args[0]

        if len(model.gpu_names) == 0:
            x = func(*args, **kwargs)
        else:
            for name in model.gpu_names:
                # TODO: Not use right now.
                # gpu_util = get_gpu_utility(int(name.split(":")[-1]))
                # assert gpu_util <= 75, "{} ({} %) >= 75%".format(name, gpu_util)
                with tf.device(name):
                    x = func(*args, **kwargs)
        return x

    return wrapped


def with_tf_device_cpu(func):
    def wrapped(*args, **kwargs):
        with tf.device("/cpu:0"):
            x = func(*args, **kwargs)
        return x

    return wrapped


def tf_session_with_config(**kwargs):
    config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        **kwargs,
    )
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# Matrix utils

def get_zero_expanded_matrix(base_matrix: np.ndarray, indexes_to_zero, add_rows=False):
    """
    :param base_matrix: np.ndarray (n, m)
    :param indexes_to_zero: np.ndarray or list of int (size k)
    :param add_rows: position to add zeros: row or column
    :return: np.ndarray (n + k, m) is_row=True or (n, m + k) is_row=False

    e.g.

    >>> get_zero_expanded_matrix(
        np.asarray([[1, 2],
                    [3, 4]]),
        [0, 2, 3],
        add_rows=True,
    )
    [[0 0]
    [1 2]
    [0 0]
    [0 0]
    [3 4]]

    >>> get_zero_expanded_matrix(
        np.asarray([[1, 2],
                    [3, 4]]),
        [0, 2, 3],
        add_rows=False,
    )
    [[0 1 0 0 2]
     [0 3 0 0 4]]
    """

    if len(base_matrix.shape) == 1:
        base_matrix = np.asarray([base_matrix])
        was_1d = True
    else:
        was_1d = False

    base_shape = base_matrix.shape
    rowwise_ret_shape = (base_shape[0] + len(indexes_to_zero), base_shape[1]) if add_rows else \
        (base_shape[1] + len(indexes_to_zero), base_shape[0])

    rowwise_base_list = list(base_matrix) if add_rows else list(np.transpose(base_matrix))
    zero_expanded_list = []
    for i in range(rowwise_ret_shape[0]):
        if i not in indexes_to_zero:
            zero_expanded_list.append(rowwise_base_list.pop(0))
        else:
            zero_expanded_list.append([0 for _ in range(rowwise_ret_shape[1])])

    zero_expanded_matrix = np.asarray(zero_expanded_list)
    zero_expanded_matrix = zero_expanded_matrix if add_rows else np.transpose(zero_expanded_matrix)

    if was_1d:
        return zero_expanded_matrix.squeeze()
    else:
        return zero_expanded_matrix


# Matplotlib utils

def build_bar(x, y, ylabel, title, draw_xticks=False, **kwargs):
    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, **kwargs)
    if draw_xticks:
        plt.xticks(y_pos, x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    return plt


def draw_importance_bar_chart(iv, prev_first_layer, curr_first_layer, prev_second_layer, task_id):
    colors = ["grey" for _ in range(len(iv))]
    for i in range(prev_first_layer, curr_first_layer):
        colors[i] = "red"
    for i in range(curr_first_layer + prev_second_layer, len(iv)):
        colors[i] = "red"
    build_bar(list(range(len(iv))), iv,
              ylabel="Importance", title="Importance of Neurons in task {}".format(task_id), color=colors)


def build_line_of_list(x_or_x_list, y_list, label_y_list, xlabel, ylabel, title, file_name,
                       highlight_ylabels=None, is_x_list=True, marker="o",
                       title_fontsize=20, label_fontsize=17, tick_fontsize="large", legend_fontsize="x-large",
                       linewidth=2.5, markersize=4,
                       **kwargs):
    if not is_x_list:
        x_or_x_list = [deepcopy(x_or_x_list) for _ in range(len(y_list))]

    for i, (x, y, yl) in enumerate(zip(x_or_x_list, y_list, label_y_list)):

        if highlight_ylabels is None:
            alpha, linestyle = 1, "-"
        elif yl in highlight_ylabels:
            alpha, linestyle = 1, "-"
        else:
            alpha, linestyle = 0.75, ":"

        plt.plot(x, y, label=yl, alpha=alpha, linewidth=linewidth, linestyle=linestyle,
                 marker=marker, markersize=markersize)

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)

    if "ylim" in kwargs:
        plt.ylim(kwargs["ylim"])

    plt.title(title, fontdict={"size": title_fontsize})
    plt.xlabel(xlabel, fontdict={"size": label_fontsize})
    plt.ylabel(ylabel, fontdict={"size": label_fontsize})
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name)

    plt.show()

    return plt


if __name__ == '__main__':
    build_line_of_list([1, 2, 3], [[1, 2, 3], [2, 2, 2]], ["a", "b"], "x", "y", "title", "file.png")
