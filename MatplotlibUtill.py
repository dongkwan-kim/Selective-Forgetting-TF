import matplotlib.pyplot as plt
import numpy as np


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
