import os
import re
from pprint import pprint
from typing import Tuple
import math

import tensorflow as tf
import numpy as np
from DEN.ops import ROC_AUC
from termcolor import cprint
from tqdm import trange

from SFN.SFNBase import SFN
from utils import get_dims_from_config, print_all_vars


def parse_pool_key(pool_name):
    p = re.compile("pool(\\d+)_ksize")
    return int(p.findall(pool_name)[0])


class SFLCL(SFN):
    """
    Selective Forgettable LeNet Variant for Large Class Learning

    LeCun et al. "Gradient-based learning applied to document recognition"
    Proceedings of the IEEE. 1998.
    """

    def __init__(self, config):
        super().__init__(config)

        self.conv_dims = get_dims_from_config(config, search="conv")
        self.pool_pos_to_dims = {parse_pool_key(k): v
                                 for k, v in get_dims_from_config(config, search="pool", with_key=True)}
        self.dims = get_dims_from_config(config, search="fc")

        self.sess = tf.Session()
        self.batch_size = config.batch_size
        self.n_layers = len(self.dims) + int(len(self.conv_dims) / 2) - 1
        self.n_classes = config.n_classes
        self.max_iter = config.max_iter
        self.init_lr = config.lr
        self.l1_lambda = config.l1_lambda
        self.l2_lambda = config.l2_lambda
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.yhat = None
        self.loss = None

        self.create_model_variables()
        self.set_layer_types()
        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")

    def create_variable(self, scope, name, shape, trainable=True) -> tf.Variable:
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable=trainable)
            self.params[w.name] = w
        return w

    def get_variable(self, scope, name, trainable=True) -> tf.Variable:
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(name, trainable=trainable)
            self.params[w.name] = w
        return w

    def get_params(self) -> dict:
        """ Access the parameters """
        params = dict()
        for scope_name, param in self.params.items():
            w = self.sess.run(param)
            params[scope_name] = w
        return params

    def load_params(self, params, *args, **kwargs):
        """ params: it contains weight parameters used in network, like ckpt """
        self.params = dict()
        for scope_name, param in params.items():
            scope_name = scope_name.split(':')[0]
            w = tf.get_variable(scope_name, initializer=param, trainable=True)
            self.params[w.name] = w
        return self.params

    def get_performance(self, p, y) -> list:
        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:, _i], y[:, _i])  # TODO: remove DEN dependency
            perf_list.append(perf)
        return [float(p) for p in perf_list]

    def predict_perform(self, xs, ys, number_to_print=6) -> list:
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        test_preds_list = []

        # (10000, 32, 32, 3) is too big, so divide to batches.
        test_batch_size = len(xs) // 5
        for i in range(5):
            partial_xs = xs[i*test_batch_size:(i+1)*test_batch_size]
            test_preds_list.append(self.sess.run(self.yhat, feed_dict={X: partial_xs}))
        test_preds = np.concatenate(test_preds_list)

        half_number_to_print = int(number_to_print / 2)
        print(" [*] Evaluation, ")
        test_perf = self.get_performance(test_preds, ys)
        for rank, idx in enumerate(reversed(np.argsort(test_perf))):
            if rank < half_number_to_print:
                print("\t Class: %s, test perf: %.4f" % (str(idx), test_perf[idx]))
            elif rank >= len(test_perf) - half_number_to_print:
                print("\t Class: %s, test perf: %.4f" % (str(idx), test_perf[idx]))

            if rank == half_number_to_print and len(test_perf) > 2 * half_number_to_print:
                print("\t ...")

        return test_perf

    def predict_only_after_training(self, **kwargs) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        _, _, _, _, test_x, test_labels = self.get_data_stream_from_task_as_class_data(shuffle=False)
        return self.predict_perform(test_x, test_labels, **kwargs)

    def set_layer_types(self):
        for i in range(2, len(self.conv_dims), 2):
            self.layer_types.append("conv")
        for i in range(1, len(self.dims)):
            self.layer_types.append("fc")

    def create_model_variables(self):
        tf.reset_default_graph()
        for i in range(2, len(self.conv_dims), 2):
            n_filters, k_size = self.conv_dims[i], self.conv_dims[i + 1]
            prev_n_filters = self.conv_dims[i - 2]
            w_conv = self.create_variable("conv%d" % (i // 2), "weight", (k_size, k_size, prev_n_filters, n_filters))
            b_conv = self.create_variable("conv%d" % (i // 2), "biases", (n_filters,))
        for i in range(1, len(self.dims)):
            prev_dims, curr_dims = self.dims[i - 1], self.dims[i]
            w_fc = self.create_variable("fc%d" % i, "weight", (prev_dims, curr_dims))
            b_fc = self.create_variable("fc%d" % i, "biases", (curr_dims,))

    def build_model(self):
        xn_filters, xsize = self.conv_dims[0], self.conv_dims[1]
        X = tf.placeholder(tf.float32, [None, xsize, xsize, xn_filters], name="X")
        Y = tf.placeholder(tf.float32, [None, self.n_classes], name="Y")

        h_conv = X
        for conv_num, i in enumerate(range(2, len(self.conv_dims), 2)):
            w_conv = self.get_variable("conv%d" % (i // 2), "weight", True)
            b_conv = self.get_variable("conv%d" % (i // 2), "biases", True)
            h_conv = tf.nn.relu(tf.nn.conv2d(h_conv, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv)

            if conv_num + 1 in self.pool_pos_to_dims:
                pool_dim = self.pool_pos_to_dims[conv_num + 1]
                pool_ksize = [1, pool_dim, pool_dim, 1]
                h_conv = tf.nn.max_pool(h_conv, ksize=pool_ksize, strides=[1, 2, 2, 1], padding="SAME")

        h_fc = tf.reshape(h_conv, (-1, self.dims[0]))
        for i in range(1, len(self.dims)):
            w_fc = self.get_variable("fc%d" % i, "weight", True)
            b_fc = self.get_variable("fc%d" % i, "biases", True)
            h_fc = tf.matmul(h_fc, w_fc) + b_fc

            if i < len(self.dims) - 1:  # Do not activate the last layer.
                h_fc = tf.nn.relu(h_fc)

        self.yhat = tf.nn.sigmoid(h_fc)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h_fc, labels=Y))

        return X, Y

    def initial_train(self, print_iter=10, *args):

        X, Y = self.build_model()

        # Add L2 loss regularizer
        l2_losses = []
        for var in tf.trainable_variables():
            if "conv" in var.name or "fc" in var.name:
                l2_losses.append(tf.nn.l2_loss(var))
        self.loss += self.l2_lambda * tf.reduce_sum(l2_losses)

        opt = tf.train.AdamOptimizer(learning_rate=self.init_lr, name="opt").minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        train_x, train_labels, val_x, val_labels, test_x, test_labels = self.get_data_stream_from_task_as_class_data()

        for epoch in range(self.max_iter):
            self.initialize_batch()
            num_batches = int(math.ceil(len(train_x) / self.batch_size))
            for _ in trange(num_batches):
                batch_x, batch_y = self.get_next_batch(train_x, train_labels)
                _, loss_val = self.sess.run([opt, self.loss], feed_dict={X: batch_x, Y: batch_y})

            if epoch % print_iter == 0 or epoch == self.max_iter - 1:
                print('\n OVERALL EVALUATION at ITERATION {}'.format(epoch))
                perfs = self.predict_perform(test_x, test_labels)
                print("   [*] avg_perf: %.4f" % float(np.mean(perfs)))

    def get_data_stream_from_task_as_class_data(self, shuffle=True, base_seed=42) -> Tuple[np.ndarray, ...]:
        """a method that combines data divided by class"""
        train_labels = np.concatenate([self.data_labels.get_train_labels(t + 1) for t in range(self.n_tasks)])
        validation_labels = np.concatenate([self.data_labels.get_validation_labels(t + 1) for t in range(self.n_tasks)])
        test_labels = np.concatenate([self.data_labels.get_test_labels(t + 1) for t in range(self.n_tasks)])

        x_and_label_list = []
        for i, (x, l) in enumerate(zip([self.trainXs, self.valXs, self.testXs],
                                       [train_labels, validation_labels, test_labels])):
            x = np.concatenate(x)
            assert len(x) == len(l)
            if shuffle:
                np.random.seed(i + base_seed)
                rand_seq = np.random.permutation(len(x))
                x_and_label_list += [x[rand_seq], l[rand_seq]]
            else:
                x_and_label_list += [x, l]

        # train_x, train_labels, val_x, validation_labels, test_x, test_labels
        return tuple(x_and_label_list)

    # shape = (|h|+|f|,) or tuple of (|f1|), (|f2|), (|h1|,), (|h2|,)
    def get_importance_vector(self, task_id, importance_criteria: str, layer_separate=False) -> tuple or np.ndarray:
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y = tf.get_default_graph().get_tensor_by_name("Y:0")

        # weight_list = []  TODO
        # bias_list = []  TODO
        hidden_layer_list = []

        h_conv = X
        for conv_num, i in enumerate(range(2, len(self.conv_dims), 2)):
            w_conv = self.get_variable("conv%d" % (i // 2), "weight", True)
            b_conv = self.get_variable("conv%d" % (i // 2), "biases", True)
            h_conv = tf.nn.relu(tf.nn.conv2d(h_conv, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv)

            print(' [*] class %d, shape of conv %d : %s' % (task_id, conv_num + 1, h_conv.get_shape().as_list()))
            hidden_layer_list.append(h_conv)  # e.g. shape = (32, 32, 64),

            if conv_num + 1 in self.pool_pos_to_dims:
                pool_dim = self.pool_pos_to_dims[conv_num + 1]
                pool_ksize = [1, pool_dim, pool_dim, 1]
                h_conv = tf.nn.max_pool(h_conv, ksize=pool_ksize, strides=[1, 2, 2, 1], padding="SAME")

        h_fc = tf.reshape(h_conv, (-1, self.dims[0]))
        for i in range(1, len(self.dims)):
            w_fc = self.get_variable("fc%d" % i, "weight", True)
            b_fc = self.get_variable("fc%d" % i, "biases", True)
            h_fc = tf.matmul(h_fc, w_fc) + b_fc

            if i < len(self.dims) - 1:  # Do not activate the last layer.
                h_fc = tf.nn.relu(h_fc)

                print(' [*] class %d, shape of fc %d : %s' % (task_id, i, h_fc.get_shape().as_list()))
                hidden_layer_list.append(h_fc)  # e.g. shape = (128,)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h_fc, labels=Y))
        _ = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)

        gradient_list = [tf.gradients(loss, h) for h in hidden_layer_list]
        h_length_list = [h.get_shape().as_list()[-1] for h in hidden_layer_list]

        return self.get_importance_vector_from_tf_vars(
            task_id, importance_criteria,
            h_length_list=h_length_list,
            hidden_layer_list=hidden_layer_list,
            gradient_list=gradient_list,
            weight_list=None,
            bias_list=None,
            X=X, Y=Y,
            layer_separate=layer_separate,
        )

    def recover_params(self, idx):
        self.assign_new_session(idx)

    def assign_new_session(self, idx_to_load_params=None):
        if idx_to_load_params is None:
            params = self.get_params()
        else:
            params = self.old_params_list[idx_to_load_params]
        self.clear()
        self.sess = tf.Session()
        self.load_params(params)
        self.sess.run(tf.global_variables_initializer())
        self.loss = None
        self.yhat = None
        self.build_model()

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        pass

    def _assign_retrained_value_to_tensor(self, task_id):
        pass

