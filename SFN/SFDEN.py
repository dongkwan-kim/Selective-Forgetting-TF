import collections
from typing import Dict, List, Callable
from termcolor import cprint

from DEN.DEN import DEN
from SFN.SFNBase import SFN

import tensorflow as tf
import numpy as np
from utils import build_line_of_list, get_zero_expanded_matrix, parse_var_name


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

    # Variable, params, ... attributes Manipulation

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

    def sfden_get_weight_and_bias_at_task(self, layer_id: int, stamp: list, task_id: int, var_prefix: str,
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

        w_key, b_key = ("weight", "biases") if not is_bottom else ("weight_%d" % task_id, "biases_%d" % task_id)
        layer_key = "layer%d" % layer_id

        w: tf.Variable = self.get_variable(layer_key, w_key, True)
        b: tf.Variable = self.get_variable(layer_key, b_key, True)
        w = w[:stamp[layer_id - 1], :stamp[layer_id]]
        b = b[:stamp[layer_id]]

        if is_forgotten:
            assert self.importance_matrix_tuple is not None

            # Remove columns (neurons in the current layer)
            removed_neurons = self.get_removed_neurons_of_layer(layer_id, stamp)
            w: np.ndarray = np.delete(w.eval(session=self.sess), removed_neurons, axis=1)
            b: np.ndarray = np.delete(b.eval(session=self.sess), removed_neurons)

            # Remove rows (neurons in the previous layer)
            if layer_id > 1:  # Do not consider 1st layer, that does not have previous layer.
                removed_neurons_prev = self.get_removed_neurons_of_layer(layer_id - 1, stamp)
                w = np.delete(w, removed_neurons_prev, axis=0)

        sfn_w = self.sfn_create_or_get_variable("%s_t%d_layer%d" % (var_prefix, task_id, layer_id), w_key,
                                                trainable=True, initializer=w)
        sfn_b = self.sfn_create_or_get_variable("%s_t%d_layer%d" % (var_prefix, task_id, layer_id), b_key,
                                                trainable=True, initializer=b)
        return sfn_w, sfn_b

    def get_removed_neurons_of_layer(self, layer_id, stamp=None) -> list:
        return [neuron for neuron in self.layer_to_removed_neuron_set["layer%d" % layer_id]
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

            if t != self.n_tasks - 1:
                self.clear()

    # Retrain after forgetting

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        """
        Note that this use sfn_get_weight_and_bias_at_task with is_forgotten=True
        """

        train_xs_t, train_labels_t, val_xs_t, val_labels_t, test_xs_t, test_labels_t = data

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            sfn_w, sfn_b = self.sfden_get_weight_and_bias_at_task(
                i, stamp, task_id, "retrain",
                is_bottom=False, is_forgotten=True)
            bottom = tf.nn.relu(tf.matmul(bottom, sfn_w) + sfn_b)
            if is_verbose:
                print(' [*] task %d, shape of layer %d : %s' % (task_id, i, sfn_w.get_shape().as_list()))

        sfn_w, sfn_b = self.sfden_get_weight_and_bias_at_task(
            self.n_layers, stamp, task_id, "retrain",
            is_bottom=True, is_forgotten=True)
        y = tf.matmul(bottom, sfn_w) + sfn_b
        yhat = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))

        train_step = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)

        loss_window = collections.deque(maxlen=10)
        old_loss = 999
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(retrain_flags.retrain_max_iter_per_task):
            self.initialize_batch()
            while True:
                batch_x, batch_y = self.get_next_batch(train_xs_t, train_labels_t)
                if len(batch_x) == 0:
                    break
                _, loss_val = self.sess.run([train_step, loss], feed_dict={X: batch_x, Y: batch_y})

            val_preds, val_loss_val = self.sess.run([yhat, loss], feed_dict={X: val_xs_t, Y: val_labels_t})
            loss_window.append(val_loss_val)
            mean_loss = np.mean(loss_window)
            val_perf = self.get_performance(val_preds, val_labels_t)

            if epoch == 0 or epoch == retrain_flags.max_iter - 1:
                if is_verbose:
                    print(" [*] iter: %d, val loss: %.4f, val perf: %.4f" % (epoch, val_loss_val, val_perf))
                if abs(old_loss - mean_loss) < 1e-6:
                    break
                old_loss = mean_loss

    def _assign_retrained_value_to_tensor(self, task_id):

        stamp = self.time_stamp['task%d' % task_id]

        # Get retrained values.
        retrained_values_dict = self.sfn_get_params(name_filter=lambda n: "_t{}_".format(task_id) in n
                                                                          and "retrain" in n)

        # get_zero_expanded_matrix.
        for name, retrained_value in list(retrained_values_dict.items()):
            prefix, _, layer_id, var_type = parse_var_name(name)
            removed_neurons = self.get_removed_neurons_of_layer(layer_id, stamp)

            # Expand columns
            retrained_value = get_zero_expanded_matrix(retrained_value, removed_neurons, add_rows=False)

            # Expand rows
            if layer_id > 1 and "weight" in var_type:
                removed_neurons_prev = self.get_removed_neurons_of_layer(layer_id - 1, stamp)
                retrained_value = get_zero_expanded_matrix(retrained_value, removed_neurons_prev, add_rows=True)

            retrained_values_dict[name] = retrained_value

        # Assign values to tensors from DEN.
        for name, retrained_value in retrained_values_dict.items():
            prefix, _, layer_id, var_type = parse_var_name(name)
            name_den = "layer{}/{}:0".format(layer_id, var_type)
            tensor_den = self.params[name_den]
            value_den = tensor_den.eval(session=self.sess)

            if "weight" in var_type:
                value_den[:stamp[layer_id - 1], :stamp[layer_id]] = retrained_value
            else:
                value_den[:stamp[layer_id]] = retrained_value

            self.sess.run(tf.assign(tensor_den, value_den))

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
            removed_neurons = self.get_removed_neurons_of_layer(layer_id, stamp)
            original_shape[1] -= len(removed_neurons)
            if layer_id > 1:
                removed_neurons_prev = self.get_removed_neurons_of_layer(layer_id - 1, stamp)
                original_shape[0] -= len(removed_neurons_prev)
            return original_shape

    def recover_params(self, idx):
        self.assign_new_session(idx)
        self.sess.run(tf.global_variables_initializer())

    # Importance vectors

    # shape = (|h|,) or tuple of (|h1|,), (|h2|,)
    def get_importance_vector(self, task_id, importance_criteria: str, layer_separate=False) -> tuple or np.ndarray:
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        hidden_layer_list = []
        weight_list = []
        bias_list = []

        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            sfn_w, sfn_b = self.sfden_get_weight_and_bias_at_task(i, stamp, task_id, "imp", is_bottom=False)
            bottom = tf.nn.relu(tf.matmul(bottom, sfn_w) + sfn_b)

            hidden_layer_list.append(bottom)
            weight_list.append(sfn_w)
            bias_list.append(sfn_b)

            print(' [*] task %d, shape of layer %d : %s' % (task_id, i, sfn_w.get_shape().as_list()))

        sfn_w, sfn_b = self.sfden_get_weight_and_bias_at_task(self.n_layers, stamp, task_id, "imp", is_bottom=True)
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
        )
