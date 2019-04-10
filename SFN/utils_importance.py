import tensorflow as tf
import numpy as np


# shape = (batch_size or 1, |h|)

def get_1st_taylor_approximation_based(sess: tf.Session, value_dict: dict, feed_dict: dict):
    # |w| * dy/dw
    # PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE, ICLR 2017
    hidden_layers, gradients = sess.run(
        [value_dict["hidden_layers"], value_dict["gradients"]],
        feed_dict=feed_dict,
    )
    return tuple(np.absolute(h * g)[0] for h, g in zip(hidden_layers, gradients))


def get_activation_based(sess: tf.Session, value_dict: dict, feed_dict: dict):
    # ReLU(x * w + b)
    # Cnvlutin: Ineffectual-Neuron-Free Deep Neural Network Computing, SIGARCH CAN 2016
    hidden_layers = sess.run(
        value_dict["hidden_layers"],
        feed_dict=feed_dict,
    )
    return tuple(hidden_layers)


def get_magnitude_based(sess: tf.Session, value_dict: dict, feed_dict: dict):
    # 1/dim(w) * \sum_{i} |w_{i}|
    # Learning both weights and connections for efficient neural network, NIPS 2015
    # Dynamic network surgery for efficient dnns, NIPS 2016

    # w: (i, o), b: (o, )
    weights, biases = sess.run(
        [value_dict["weights"], value_dict["biases"]],
        feed_dict=feed_dict,
    )
    return tuple((np.sum(w, axis=0) + b) / (w.shape[-1] + 1) for w, b in zip(weights, biases))


def get_gradient_based(sess: tf.Session, value_dict: dict, feed_dict: dict):
    # Learning Sparse Neural Networks via Sensitivity-Driven Regularization, NIPS 2018
    gradients = sess.run(
        value_dict["gradients"],
        feed_dict=feed_dict,
    )
    return tuple(g[0] for g in gradients)


def get_hessian_based(sess: tf.Session, value_dict: dict, feed_dict: dict):
    # 1/2 * w_{i}^2 * H_{ii}
    # Optimal brain damage, NIPS 1990
    # Optimal Brain Surgeon and general network pruning, NN 1993
    raise NotImplementedError
