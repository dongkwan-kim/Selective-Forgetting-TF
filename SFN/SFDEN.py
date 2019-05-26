import collections
from typing import Dict, List, Callable
from termcolor import cprint
import math
import re

from DEN.DEN import DEN

from SFNBase import SFN

import tensorflow as tf
import numpy as np
from utils import build_line_of_list, get_zero_expanded_matrix, parse_var_name, print_all_vars, with_tf_device_cpu, \
    with_tf_device_gpu


class SFDEN(DEN, SFN):
    """
    Selective Forgettable Dynamic Expandable Network

    Yoon et al. "Lifelong Learning with Dynamically Expandable Networks"
    International Conference on Learning Representations}. 2018.
    """

    def __init__(self, config):
        SFN.__init__(self, config)
        DEN.__init__(self, config)
        self.attr_to_save += ["T", "time_stamp"]
        self.set_layer_types()

        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")
        cprint("Device info: {}".format(self.get_real_device_info()), "green")

    # Variable, params, ... attributes Manipulation

    def set_layer_types(self):
        for i in range(self.n_layers - 1):
            self.layer_types.append("layer")
        self.layer_types.append("layer")

    def create_model_variables(self):
        tf.reset_default_graph()
        last_task_dims = self.time_stamp["task{}".format(self.n_tasks)]
        for i in range(self.n_layers - 1):
            self.create_variable('layer%d' % (i + 1), 'weight', [last_task_dims[i], last_task_dims[i + 1]])
            self.create_variable('layer%d' % (i + 1), 'biases', [last_task_dims[i + 1]])

        for task_id in range(1, self.n_tasks + 1):
            self.create_variable('layer%d' % self.n_layers, 'weight_%d' % task_id,
                                 [last_task_dims[-2], self.n_classes], True)
            self.create_variable('layer%d' % self.n_layers, 'biases_%d' % task_id, [self.n_classes], True)

    def assign_new_session(self, idx=None):
        if idx is None:
            params = self.get_params()
        else:
            params = self.old_params_list[idx]
        self.clear()
        self.sess = tf.Session()
        self.load_params(params)

    def get_variables_for_sfden(self, layer_id: int, stamp: list, task_id: int, var_prefix: str,
                                is_bottom=False, is_forgotten=False):
        """
        :param layer_id: id of layer 1 ~
        :param stamp: time_stamp
        :param task_id: id of task 1 ~
        :param var_prefix: variable prefix
        :param is_bottom: whether it is bottom layer (bottom layer differs by task)
        :param is_forgotten: whether it is forgotten neural layers (delete rows & col that represent forgotten neurons)
        :return: tuple of w and b

        e.g. when layer_id == l,

        layer (l-1)th, weight whose shape is (r0, c0), bias whose shape is (b0,)
        layer (l)th, weight whose shape is (r1, c1), bias whose shape is (b1,)
        removed_neurons of (l-1)th [...] whose length is n0
        removed_neurons of (l)th [...] whose length is n1

        return weight whose shape is (r1 - n0, c1 - n1)
               bias whose shape is (b1 - n1)
        """

        weight_name, bias_name = ("weight", "biases") if not is_bottom \
            else ("weight_%d" % task_id, "biases_%d" % task_id)
        scope = "layer%d" % layer_id

        w: tf.Variable = self.get_variable(scope, weight_name, True)
        b: tf.Variable = self.get_variable(scope, bias_name, True)
        w = w[:stamp[layer_id - 1], :stamp[layer_id]]
        b = b[:stamp[layer_id]]

        if is_forgotten:
            return self.get_retraining_vars_from_old_vars(
                scope=scope, weight_name=weight_name, bias_name=bias_name,
                task_id=task_id, var_prefix=var_prefix,
                weight_var=w, bias_var=b,
                stamp=stamp,
            )
        else:
            sfn_w = self.sfn_create_or_get_variable("%s_t%d_layer%d" % (var_prefix, task_id, layer_id), weight_name,
                                                    trainable=True, initializer=w)
            sfn_b = self.sfn_create_or_get_variable("%s_t%d_layer%d" % (var_prefix, task_id, layer_id), bias_name,
                                                    trainable=True, initializer=b)
            return sfn_w, sfn_b

    def get_removed_neurons_of_scope(self, scope, stamp=None) -> list:
        layer_id = int(re.findall(r'\d+', scope)[0])
        return [neuron for neuron in self.layer_to_removed_unit_set[scope]
                if stamp is None or neuron < stamp[layer_id]]

    # Train DEN: code from github.com/dongkwan-kim/DEN/blob/master/DEN/DEN_run.py

    def initial_train(self):
        params = dict()
        avg_perf = []

        for t in range(self.n_tasks):
            data = (self.trainXs[t], self.data_labels.get_train_labels(t + 1),
                    self.valXs[t], self.data_labels.get_validation_labels(t + 1),
                    self.testXs[t], self.data_labels.get_test_labels(t + 1))

            self.sess = tf.Session()

            print("\n\n\tTASK %d TRAINING\n" % (t + 1))
            self.task_inc()
            self.load_params(params, time=1)
            perf, sparsity, expansion = self.add_task(t + 1, data)

            # Do not use self.assign_new_session() in this code block. I do not why but it does not work.
            params = self.get_params()
            self.clear()
            self.sess = tf.Session()
            self.load_params(params)

            print('\n OVERALL EVALUATION')
            overall_perfs = []
            for j in range(t + 1):
                temp_perf = self.predict_perform(j + 1, self.testXs[j], self.data_labels.get_test_labels(j + 1))
                overall_perfs.append(temp_perf)
            avg_perf.append(sum(overall_perfs) / float(t + 1))
            print("   [*] avg_perf: %.4f" % avg_perf[t])

            if self.online_importance:
                self.save_online_importance_matrix(t + 1)

            if t != self.n_tasks - 1:
                self.clear()

    # Retrain after forgetting

    def _retrain_at_task_or_all(self, task_id, train_xs, train_labels, retrain_flags, is_verbose, **kwargs):
        """
        Note that this use sfn_get_weight_and_bias_at_task with is_forgotten=True
        """

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])
        keep_prob = tf.placeholder(tf.float32)

        variables = []
        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            sfn_w, sfn_b = self.get_variables_for_sfden(
                i, stamp, task_id, "retrain",
                is_bottom=False, is_forgotten=True)
            variables += [sfn_w, sfn_b]
            bottom = tf.nn.relu(tf.matmul(bottom, sfn_w) + sfn_b)
            bottom = tf.nn.dropout(bottom, keep_prob=keep_prob)
            if is_verbose:
                print(' [*] task %d, shape of layer %d : %s' % (task_id, i, sfn_w.get_shape().as_list()))

        sfn_w, sfn_b = self.get_variables_for_sfden(
            self.n_layers, stamp, task_id, "retrain",
            is_bottom=True, is_forgotten=True)
        variables += [sfn_w, sfn_b]

        y = tf.matmul(bottom, sfn_w) + sfn_b
        yhat = tf.nn.sigmoid(y)

        l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0, scale_l2=0.00001)
        regularization_loss = tf.contrib.layers.apply_regularization(l1_l2_regularizer, variables)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y)) + regularization_loss

        train_step = tf.train.AdamOptimizer(self.init_lr).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        _, loss_val = self.sess.run([train_step, loss],
                                    feed_dict={X: train_xs, Y: train_labels, keep_prob: 0.9})
        print("  [*] loss(t{}): {}".format(task_id, loss_val))

    def _assign_retrained_value_to_tensor(self, task_id, **kwargs):

        stamp = self.time_stamp['task%d' % task_id]

        def _value_preprocess_sfden(name, value_sfn, retrained_value) -> np.ndarray:
            _, _, scope, var_type = parse_var_name(name)
            layer_id = int(re.findall(r'\d+', scope)[0])
            if "weight" in var_type:
                value_sfn[:stamp[layer_id - 1], :stamp[layer_id]] = retrained_value
            else:
                value_sfn[:stamp[layer_id]] = retrained_value
            return value_sfn

        super()._assign_retrained_value_to_tensor(
            task_id,
            value_preprocess=_value_preprocess_sfden,
            stamp=stamp,
        )

    def predict_only_after_training(self) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        temp_perfs = []
        for t in range(self.n_tasks):
            temp_perf = self.predict_perform(t + 1, self.testXs[t], self.data_labels.get_test_labels(t + 1))
            temp_perfs.append(temp_perf)
        return temp_perfs

    def get_shape(self, obj_w_shape, task_id, layer_id):
        """Override original get_shape"""
        if not self.retrained:
            return super().get_shape(obj_w_shape, task_id, layer_id)
        else:
            original_shape = super().get_shape(obj_w_shape, task_id, layer_id)
            stamp = self.time_stamp['task%d' % task_id]
            removed_neurons = self.get_removed_neurons_of_scope("layer%d" % layer_id, stamp)
            original_shape[1] -= len(removed_neurons)
            if layer_id > 1:
                removed_neurons_prev = self.get_removed_neurons_of_scope("layer%d" % (layer_id - 1), stamp)
                original_shape[0] -= len(removed_neurons_prev)
            return original_shape

    def recover_params(self, idx):
        self.assign_new_session(idx)
        self.sess.run(tf.global_variables_initializer())

    # Importance vectors

    # shape = (|h|,) or tuple of (|h1|,), (|h2|,)
    def get_importance_vector(self, task_id, importance_criteria: str,
                              layer_separate=False, use_coreset=False) -> tuple or np.ndarray:
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        hidden_layer_list = []
        weight_list = []
        bias_list = []

        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            sfn_w, sfn_b = self.get_variables_for_sfden(i, stamp, task_id, "imp", is_bottom=False)
            bottom = tf.nn.relu(tf.matmul(bottom, sfn_w) + sfn_b)

            hidden_layer_list.append(bottom)
            weight_list.append(sfn_w)
            bias_list.append(sfn_b)

            print(' [*] task %d, shape of layer %d : %s' % (task_id, i, sfn_w.get_shape().as_list()))

        sfn_w, sfn_b = self.get_variables_for_sfden(self.n_layers, stamp, task_id, "imp", is_bottom=True)
        y = tf.matmul(bottom, sfn_w) + sfn_b
        yhat = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))

        _ = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)
        gradient_list = [tf.gradients(loss, h) for h in hidden_layer_list]

        self.sess.run(tf.global_variables_initializer())

        h_length_list = [h.get_shape().as_list()[-1] for h in hidden_layer_list]

        # layer_separate = True: tuple of ndarray of shape (|h1|,), (|h2|,) or
        # layer_separate = False: ndarray of shape (|h|,)
        return self.get_importance_vector_from_tf_vars(
            task_id, importance_criteria,
            h_length_list=h_length_list,
            hidden_layer_list=hidden_layer_list,
            gradient_list=gradient_list,
            weight_list=weight_list,
            bias_list=bias_list,
            X=X, Y=Y,
            layer_separate=layer_separate,
            use_coreset=use_coreset,
        )
