from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import re


def sum_set(s: set, *args):
    s = deepcopy(s)
    for _s in args:
        s.update(_s)
    return s


def parse_var_name(var_name):
    p = re.compile("(.+)_t(\d)_layer(\d)/(.+):0")
    return [x if not x.isnumeric() else int(x) for x in p.findall(var_name)[0]]


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


def build_line_of_list(x, y_list, label_y_list, xlabel, ylabel, title, file_name,
                       highlight_yi=None, **kwargs):

    for i, (y, yl) in enumerate(zip(y_list, label_y_list)):

        if highlight_yi is None:
            alpha, linewidth = 1, 1
        elif highlight_yi == i:
            alpha, linewidth = 1, 3
        else:
            alpha, linewidth = 0.75, 1

        plt.plot(x, y, label=yl, alpha=alpha, linewidth=linewidth)

    plt.legend()

    if "ylim" in kwargs:
        plt.ylim(kwargs["ylim"])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if file_name:
        plt.savefig(file_name)

    plt.show()

    return plt


if __name__ == '__main__':
    build_line_of_list([1, 2, 3], [[1, 2, 3], [2, 2, 2]], ["a", "b"], "x", "y", "title", "file.png")
