import tensorflow as tf
import numpy as np


def get_gradient_times_activation(hidden_layer, gradient):
    return np.absolute(hidden_layer * gradient)[0]

