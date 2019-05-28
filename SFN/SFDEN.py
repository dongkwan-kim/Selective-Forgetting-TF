import collections
from typing import Dict, List, Callable
from termcolor import cprint
import math
import re

from DEN.DEN import DEN

from SFNBase import SFN
from enums import MaskType
from mask import Mask

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

        if is_forgotten:  # TODO: remove legacy
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

    def get_retraining_vars_from_old_vars(self, scope: str,
                                          weight_name: str, bias_name: str,
                                          task_id: int, var_prefix: str,
                                          weight_var=None, bias_var=None, **kwargs) -> tuple:

        """
        :param scope: scope of variables.
        :param weight_name: if weight_variable is None, use get_variable else use it.
        :param bias_name: if bias_variable is None, use get_variable else use it.
        :param task_id: id of task / 1 ~
        :param var_prefix: variable prefix
        :param weight_var: weight_variable
        :param bias_var: bias_variable
        :param kwargs: kwargs for get_removed_neurons_of_scope
        :return: tuple of w and b

        e.g. when scope == "layer2",
        layer-"layer2", weight whose shape is (r0, c0), bias whose shape is (b0,)
        layer-next of "layer2", weight whose shape is (r1, c1), bias whose shape is (b1,)
        removed_neurons of (l-1)th [...] whose length is n0
        removed_neurons of (l)th [...] whose length is n1
        return weight whose shape is (r1 - n0, c1 - n1)
               bias whose shape is (b1 - n1)
        """
        scope_list = self._get_scope_list()
        scope_idx = scope_list.index(scope)
        prev_scope = scope_list[scope_idx - 1] if scope_idx > 0 else None

        assert self.importance_matrix_tuple is not None
        w: tf.Variable = self.get_variable(scope, weight_name, True) \
            if weight_var is None else weight_var
        b: tf.Variable = self.get_variable(scope, bias_name, True) \
            if bias_var is None else bias_var

        # TODO: CNN

        # Remove columns (neurons in the current layer)
        removed_neurons = self.get_removed_neurons_of_scope(scope, **kwargs)
        w: np.ndarray = np.delete(w.eval(session=self.sess), removed_neurons, axis=1)
        b: np.ndarray = np.delete(b.eval(session=self.sess), removed_neurons)

        # Remove rows (neurons in the previous layer)
        if prev_scope:  # Do not consider 1st layer, that does not have previous layer.
            removed_neurons_prev = self.get_removed_neurons_of_scope(prev_scope, **kwargs)
            w = np.delete(w, removed_neurons_prev, axis=0)

        sfn_w = self.sfn_create_or_get_variable("%s_t%d_%s" % (var_prefix, task_id, scope), weight_name,
                                                trainable=True, initializer=w)
        sfn_b = self.sfn_create_or_get_variable("%s_t%d_%s" % (var_prefix, task_id, scope), bias_name,
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

    def build_model_for_retraining(self, flags):
        X_list = [tf.placeholder(tf.float32, [None, self.dims[0]], name="X_{}".format(t))
                  for t in range(self.n_tasks)]
        Y_list = [tf.placeholder(tf.float32, [None, self.n_classes], name="Y_{}".format(t))
                  for t in range(self.n_tasks)]
        yhat_list = []
        loss_list = []

        for t, (X, Y) in enumerate(zip(X_list, Y_list)):

            task_id = t + 1

            bottom = X
            stamp = self.time_stamp['task%d' % task_id]

            for i in range(1, self.n_layers):
                w: tf.Variable = self.get_variable("layer{}".format(i), "weight", True)
                b: tf.Variable = self.get_variable("layer{}".format(i), "biases", True)
                w = w[:stamp[i - 1], :stamp[i]]
                b = b[:stamp[i]]

                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

                ndarray_hard_mask = np.ones(shape=bottom.get_shape().as_list()[-1:])
                indices_to_zero = np.asarray(list(self.layer_to_removed_unit_set["layer{}".format(i)]))
                indices_to_zero = indices_to_zero[indices_to_zero < len(ndarray_hard_mask)]
                ndarray_hard_mask[indices_to_zero] = 0
                tf_hard_mask = tf.constant(ndarray_hard_mask, dtype=tf.float32)

                bottom = Mask(
                    i - 1, 0, 0, mask_type=MaskType.HARD, hard_mask=tf_hard_mask
                ).get_masked_tensor(bottom)

            w = self.get_variable("layer{}".format(self.n_layers), "weight_{}".format(task_id), True)
            b = self.get_variable("layer{}".format(self.n_layers), "biases_{}".format(task_id), True)
            w = w[:stamp[self.n_layers - 1], :]
            y = tf.matmul(bottom, w) + b

            yhat_list.append(tf.nn.softmax(y))

            if task_id not in flags.task_to_forget:
                loss_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=Y)))

        l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0, scale_l2=1e-8)
        regularization_loss = tf.contrib.layers.apply_regularization(
            l1_l2_regularizer,
            [v for v in tf.trainable_variables() if "weight" in v.name or "biases" in v.name]
        )

        loss = sum(loss_list) + regularization_loss
        opt = tf.train.AdamOptimizer(self.init_lr).minimize(loss)
        self.sess.run(tf.global_variables_initializer())

        return X_list, Y_list, yhat_list, opt, loss

    def _retrain_at_task_or_all(self, task_id, train_xs, train_labels, retrain_flags, is_verbose, **kwargs):
        X_list, Y_list, yhat_list, opt, loss = kwargs["model_args"]
        x_feed_dict = {X: train_xs[t]() for t, X in enumerate(X_list) if t + 1 not in retrain_flags.task_to_forget}
        y_feed_dict = {Y: train_labels[t]() for t, Y in enumerate(Y_list) if t + 1 not in retrain_flags.task_to_forget}
        _, loss_val = self.sess.run([opt, loss], feed_dict={**x_feed_dict, **y_feed_dict})
        return loss_val

    def predict_only_after_training(self, refresh_session=True, **kwargs) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        temp_perfs = []
        for t in range(self.n_tasks):
            if refresh_session:
                temp_perf = self.predict_perform(t + 1, self.testXs[t], self.data_labels.get_test_labels(t + 1))
            else:
                temp_perf = self.predict_perform_with_current_graph(
                    t + 1, self.testXs[t], self.data_labels.get_test_labels(t + 1), **kwargs)
            temp_perfs.append(temp_perf)
        print("   [*] avg_perf: %.4f +- %.4f" % (float(np.mean(temp_perfs)), float(np.std(temp_perfs))))
        return temp_perfs

    def predict_perform_with_current_graph(self, task_id, xs, ys, **kwargs):
        X_list, Y_list, yhat_list, opt, loss = kwargs["model_args"]
        test_preds = self.sess.run(yhat_list[task_id - 1], feed_dict={X_list[task_id - 1]: xs})
        test_perf = self.get_performance(test_preds, ys)
        print(" [*] Evaluation, Task:%s, test_acc: %.4f" % (str(task_id), test_perf))
        return test_perf

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
