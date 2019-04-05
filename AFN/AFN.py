import collections
from collections import defaultdict
from typing import Dict, List, Callable
import os
import pickle
from pprint import pprint
from termcolor import cprint

from DEN.DEN import DEN
from DEN.utils import print_all_vars

from data import PermutedMNISTCoreset
from utils import build_line_of_list, get_zero_expanded_matrix, parse_var_name
from utils_importance import *
from AFNBO import plot_gp
from bayes_opt import BayesianOptimization


class AFN(DEN):

    def __init__(self, config):
        super().__init__(config)
        self.n_tasks = config.n_tasks
        self.afn_params = {}
        self.batch_idx = 0
        self.mnist, self.trainXs, self.valXs, self.testXs = None, None, None, None

        self.importance_matrix_tuple = None
        self.importance_criteria = config.importance_criteria

        self.old_params_list = []
        self.prediction_history: Dict[str, List] = defaultdict(list)
        self.layer_to_removed_neuron_set: Dict[str, set] = defaultdict(set)

        self.retrained = False

        self.attr_to_save = [
            "importance_matrix_tuple",
            "importance_criteria",
            "old_params_list",
            "layer_to_removed_neuron_set",
            "time_stamp",
            "n_tasks",
            "T",
        ]

    def __repr__(self):
        return "{}_{}_{}".format(self.__class__.__name__, self.n_tasks, "_".join(map(str, self.dims)))

    # Save & Restore

    def save(self, model_name=None):
        model_name = model_name or str(self)
        model_path = os.path.join(self.checkpoint_dir, "{}.ckpt".format(model_name))

        # Model Save
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)
        print_all_vars("Saved: {}".format(model_path), "blue")

        # Attribute Save
        self.save_attr(model_name)

    def save_attr(self, model_name=None, attr=None):
        model_name = model_name or str(self)
        attr_path = os.path.join(self.checkpoint_dir, "{}_attr.pkl".format(model_name))
        attr = attr or self.attr_to_save
        with open(attr_path, "wb") as f:
            pickle.dump({k: self.__dict__[k] for k in attr}, f)
        cprint("Saved: attribute of {}".format(model_name), "blue")
        for a in attr:
            print("\t - {}".format(a))

    def restore(self, model_name=None):
        model_name = model_name or str(self)
        model_path = os.path.join(self.checkpoint_dir, "{}.ckpt".format(model_name))

        if not os.path.isfile("{}.meta".format(model_path)):
            return False

        try:
            # Attribute Restore
            self.restore_attr(model_name)

            # Model Reconstruction
            tf.reset_default_graph()
            last_task_dims = self.time_stamp["task{}".format(self.n_tasks)]
            for i in range(self.n_layers - 1):
                self.create_variable('layer%d' % (i + 1), 'weight', [last_task_dims[i], last_task_dims[i + 1]])
                self.create_variable('layer%d' % (i + 1), 'biases', [last_task_dims[i + 1]])

            for task_id in range(1, self.n_tasks + 1):
                self.create_variable('layer%d' % self.n_layers, 'weight_%d' % task_id,
                                     [last_task_dims[-2], self.n_classes], True)
                self.create_variable('layer%d' % self.n_layers, 'biases_%d' % task_id, [self.n_classes], True)

            # Model Restore
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            print_all_vars("Restored: {}".format(model_path), "blue")
            return True

        except Exception as e:
            print("Restore Failed: {}".format(model_path), str(e))
            return False

    def restore_attr(self, model_name=None):
        model_name = model_name or str(self)
        attr_path = os.path.join(self.checkpoint_dir, "{}_attr.pkl".format(model_name))
        with open(attr_path, "rb") as f:
            attr_dict: dict = pickle.load(f)
            self.__dict__.update(attr_dict)
        cprint("Restored: attribute of {}".format(model_name), "blue")
        for a in attr_dict.keys():
            print("\t - {}".format(a))

    def add_dataset(self, mnist, trainXs, valXs, testXs):
        self.mnist, self.trainXs, self.valXs, self.testXs = mnist, trainXs, valXs, testXs

    # Variable, params, ...attributes Manipulation

    def afn_create_variable(self, scope, name, shape=None, trainable=True, initializer=None):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
            if 'new' not in w.name:
                self.afn_params[w.name] = w
        return w

    def afn_get_variable(self, scope, name, trainable=True):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(name, trainable=trainable)
            self.afn_params[w.name] = w
        return w

    def afn_create_or_get_variable(self, scope, name, shape=None, trainable=True, initializer=None):
        try:
            w = self.afn_create_variable(scope, name, shape, trainable, initializer)
        except ValueError:
            w = self.afn_get_variable(scope, name, trainable)
        return w

    def afn_get_weight_and_bias_at_task(self, layer_id: int, stamp: list, task_id: int, var_prefix: str,
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

        afn_w = self.afn_create_or_get_variable("%s_t%d_layer%d" % (var_prefix, task_id, layer_id), w_key,
                                                trainable=True, initializer=w)
        afn_b = self.afn_create_or_get_variable("%s_t%d_layer%d" % (var_prefix, task_id, layer_id), b_key,
                                                trainable=True, initializer=b)
        return afn_w, afn_b

    def afn_get_params(self, name_filter: Callable = None):
        """ Access the afn_parameters """
        mdict = dict()
        for scope_name, param in self.afn_params.items():
            if name_filter is None or name_filter(scope_name):
                w = self.sess.run(param)
                mdict[scope_name] = w
        return mdict

    def get_removed_neurons_of_layer(self, layer_id, stamp=None) -> list:
        return [neuron for neuron in self.layer_to_removed_neuron_set["layer%d" % layer_id]
                if stamp is None or neuron < stamp[layer_id]]

    def clear(self):
        self.destroy_graph()
        self.sess.close()

    # Train DEN: code from github.com/dongkwan-kim/DEN/blob/master/DEN/DEN_run.py

    def train_den(self, flags):
        params = dict()
        avg_perf = []

        for t in range(flags.n_tasks):
            data = (self.trainXs[t], self.mnist.train.labels,
                    self.valXs[t], self.mnist.validation.labels,
                    self.testXs[t], self.mnist.test.labels)

            self.sess = tf.Session()

            print("\n\n\tTASK %d TRAINING\n" % (t + 1))
            self.task_inc()
            self.load_params(params, time=1)
            perf, sparsity, expansion = self.add_task(t + 1, data)

            print('\n OVERALL EVALUATION')
            params = self.get_params()
            self.clear()
            self.sess = tf.Session()
            self.load_params(params)
            temp_perfs = []
            for j in range(t + 1):
                temp_perf = self.predict_perform(j + 1, self.testXs[j], self.mnist.test.labels)
                temp_perfs.append(temp_perf)
            avg_perf.append(sum(temp_perfs) / float(t + 1))
            print("   [*] avg_perf: %.4f" % avg_perf[t])

            if t != flags.n_tasks - 1:
                self.clear()

    # Retrain after forgetting

    def retrain_after_forgetting(self, flags, policy, coreset: PermutedMNISTCoreset = None):
        cprint("\n RETRAIN AFTER FORGETTING - {}".format(policy), "green")
        self.retrained = True

        for t in range(flags.n_tasks):

            if (t + 1) != flags.task_to_forget:
                coreset_t = coreset[t] if coreset is not None else (self.trainXs[t], self.mnist.train.labels,
                                                                    self.valXs[t], self.mnist.validation.labels,
                                                                    self.testXs[t], self.mnist.test.labels)

                cprint("\n\n\tTASK %d RE-TRAINING\n" % (t + 1), "green")
                self._retrain_at_task(t + 1, coreset_t, flags)
                self._assign_retrained_value_to_tensor(t + 1)

                params = self.get_params()
                self.clear()
                self.sess = tf.Session()
                self.load_params(params)
                self.predict_only_after_training()

            else:
                cprint("\n\n\tTASK %d NO NEED TO RE-TRAINING" % (t + 1), "green")

    def _retrain_at_task(self, task_id, data, retrain_flags):
        """
        Note that this use afn_get_weight_and_bias_at_task with is_forgotten=True
        """

        train_xs_t, train_labels_t, val_xs_t, val_labels_t, test_xs_t, test_labels_t = data

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            afn_w, afn_b = self.afn_get_weight_and_bias_at_task(
                i, stamp, task_id, "retrain",
                is_bottom=False, is_forgotten=True)
            bottom = tf.nn.relu(tf.matmul(bottom, afn_w) + afn_b)
            print(' [*] task %d, shape of layer %d : %s' % (task_id, i, afn_w.get_shape().as_list()))

        afn_w, afn_b = self.afn_get_weight_and_bias_at_task(
            self.n_layers, stamp, task_id, "retrain",
            is_bottom=True, is_forgotten=True)
        y = tf.matmul(bottom, afn_w) + afn_b
        yhat = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))

        train_step = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)

        loss_window = collections.deque(maxlen=10)
        print_iter, old_loss = int(retrain_flags.max_iter / 3), 999
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(retrain_flags.max_iter):
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

            if epoch == 0 or epoch == retrain_flags.max_iter - 1 or (epoch + 1) % print_iter == 0:
                print(" [*] iter: %d, val loss: %.4f, val perf: %.4f" % (epoch, val_loss_val, val_perf))
                if abs(old_loss - mean_loss) < 1e-6:
                    break
                old_loss = mean_loss

    def _assign_retrained_value_to_tensor(self, task_id):

        stamp = self.time_stamp['task%d' % task_id]

        # Get retrained values.
        retrained_values_dict = self.afn_get_params(name_filter=lambda n: "t{}".format(task_id) in n and "retrain" in n)

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

    def predict_only_after_training(self):
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        temp_perfs = []
        for t in range(self.T):
            temp_perf = self.predict_perform(t + 1, self.testXs[t], self.mnist.test.labels)
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

    # Data batch

    def initialize_batch(self):
        self.batch_idx = 0

    def get_next_batch(self, x, y, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size
        next_idx = self.batch_idx + batch_size
        r = x[self.batch_idx:next_idx], y[self.batch_idx:next_idx]
        self.batch_idx = next_idx
        return r

    # Utils for sequential experiments

    def clear_experiments(self):
        self.layer_to_removed_neuron_set = defaultdict(set)
        self.recover_old_params()

    def recover_recent_params(self):
        print("\n RECOVER RECENT PARAMS")
        self.recover_params(-1)

    def recover_old_params(self):
        print("\n RECOVER OLD PARAMS")
        self.recover_params(0)

    def recover_params(self, idx):
        self.params = self.old_params_list[idx]
        self.clear()
        self.sess = tf.Session()
        self.load_params(self.params)
        self.sess.run(tf.global_variables_initializer())

    # Data visualization
    # TODO: use pycm

    def print_history(self, one_step_neuron=1):
        for policy, history in self.prediction_history.items():
            print("\t".join([policy] + [str(x) for x in range(1, len(history[0]) + 1)]))
            for i, acc in enumerate(history):
                print("\t".join([str(i * one_step_neuron)] + [str(x) for x in acc]))

    def print_summary(self, task_id, one_step_neuron=1):
        for policy, history in self.prediction_history.items():
            print("\t".join([policy] + [str(x) for x in range(1, len(history[0]) + 1)] + ["Acc-{}".format(policy)]))
            for i, acc in enumerate(history):
                acc_except_t = np.delete(acc, task_id - 1)
                mean_acc = np.mean(acc_except_t)
                print("\t".join([str(i * one_step_neuron)] + [str(x) for x in acc] + [str(mean_acc)]))

    def draw_chart_summary(self, task_id, one_step_neuron=1, file_prefix=None, file_extension=".png", ylim=None):

        mean_acc_except_t = None
        x_removed_neurons = None

        for policy, history in self.prediction_history.items():

            x_removed_neurons = [i * one_step_neuron for i, acc in enumerate(history)]
            history_txn = np.transpose(history)
            tasks = [x for x in range(1, self.T + 1)]

            build_line_of_list(x=x_removed_neurons, y_list=history_txn, label_y_list=tasks,
                               xlabel="Removed Neurons", ylabel="Accuracy",
                               ylim=ylim or [0.5, 1],
                               title="Accuracy by {} Neuron Deletion".format(policy),
                               file_name="{}_{}{}".format(file_prefix, policy, file_extension),
                               highlight_yi=task_id - 1)

            history_txn_except_t = np.delete(history_txn, task_id - 1, axis=0)
            history_n_mean_except_t = np.mean(history_txn_except_t, axis=0)

            if mean_acc_except_t is None:
                mean_acc_except_t = history_n_mean_except_t
            else:
                mean_acc_except_t = np.vstack((mean_acc_except_t, history_n_mean_except_t))

        build_line_of_list(x=x_removed_neurons, y_list=mean_acc_except_t,
                           label_y_list=[policy for policy in self.prediction_history.keys()],
                           xlabel="Removed Neurons", ylabel="Mean Accuracy",
                           ylim=ylim or [0.7, 1],
                           title="Mean Accuracy Except Forgetting Task-{}".format(task_id),
                           file_name="{}_MeanAcc{}".format(file_prefix, file_extension))

    # Adaptive forgetting

    def adaptive_forget(self, task_to_forget, number_of_neurons, policy):
        assert policy in ["BAL_MIX", "MIX", "VAR", "EIN", "LIN", "BAL_LIN", "RANDOM", "ALL", "ALL_VAR"]

        cprint("\n ADAPTIVE FORGET {} task-{} from {}, neurons-{}".format(
            policy, task_to_forget, self.T, number_of_neurons), "green")

        self.old_params_list.append(self.get_params())

        if not self.importance_matrix_tuple:
            self.get_importance_matrix()

        if policy == "MIX":
            ni_1, ni_2 = self.get_neurons_by_mixed_ein_and_lin(task_to_forget, number_of_neurons)
        elif policy == "BAL_MIX":
            ni_1, ni_2 = self.get_neurons_by_mixed_ein_and_lin(task_to_forget, number_of_neurons, balanced=True)
        elif policy == "VAR":
            ni_1, ni_2 = self.get_neurons_with_task_variance(task_to_forget, number_of_neurons)
        elif policy == "EIN":
            ni_1, ni_2 = self.get_exceptionally_important_neurons_for_t(task_to_forget, number_of_neurons)
        elif policy == "LIN":
            ni_1, ni_2 = self.get_least_important_neurons_for_others(task_to_forget, number_of_neurons)
        elif policy == "BAL_LIN":
            ni_1, ni_2 = self.get_least_important_neurons_for_others(task_to_forget, number_of_neurons, balanced=True)
        elif policy == "RANDOM":
            ni_1, ni_2 = self.get_random_neurons(number_of_neurons)
        elif policy == "ALL":
            ni_1, ni_2 = self.get_least_important_neurons_for_others([], number_of_neurons)
        elif policy == "ALL_VAR":
            ni_1, ni_2 = self.get_neurons_with_task_variance([], number_of_neurons)
        else:
            raise NotImplementedError

        self._remove_neurons("layer1", ni_1)
        self._remove_neurons("layer2", ni_2)

        params = self.get_params()
        self.clear()
        self.sess = tf.Session()
        self.load_params(params)

    def sequentially_adaptive_forget_and_predict(self, task_to_forget, one_step_neurons, steps, policy):

        cprint("\n SEQUENTIALLY ADAPTIVE FORGET {} task-{} from {}, neurons-{}".format(
            policy, task_to_forget, self.T, one_step_neurons * steps), "green")

        for i in range(steps + 1):
            self.adaptive_forget(task_to_forget, i * one_step_neurons, policy)
            pred = self.predict_only_after_training()
            self.prediction_history[policy].append(pred)

            if i != steps:
                self.recover_recent_params()

    def _remove_neurons(self, scope, indexes: np.ndarray):
        """Zeroing columns of target indexes"""

        if len(indexes) == 0:
            return

        print("\n REMOVE NEURONS {} ({}) - {}".format(scope, len(indexes), indexes))
        self.layer_to_removed_neuron_set[scope].update(set(indexes))

        w: tf.Variable = self.get_variable(scope, "weight", False)
        b: tf.Variable = self.get_variable(scope, "biases", False)

        val_w = w.eval(session=self.sess)
        val_b = b.eval(session=self.sess)

        for i in indexes:
            val_w[:, i] = 0
            val_b[i] = 0

        self.sess.run(tf.assign(w, val_w))
        self.sess.run(tf.assign(b, val_b))

        self.params[w.name] = w
        self.params[b.name] = b

    # Importance vectors

    # shape = (|h|,) or tuple of (|h1|,), (|h2|,)
    def get_importance_vector(self, task_id, importance_criteria: str, layer_separate=False):
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        hidden_layer_list = []
        weight_list = []
        bias_list = []

        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            afn_w, afn_b = self.afn_get_weight_and_bias_at_task(i, stamp, task_id, "imp", is_bottom=False)
            bottom = tf.nn.relu(tf.matmul(bottom, afn_w) + afn_b)

            hidden_layer_list.append(bottom)
            weight_list.append(afn_w)
            bias_list.append(afn_b)

            print(' [*] task %d, shape of layer %d : %s' % (task_id, i, afn_w.get_shape().as_list()))

        afn_w, afn_b = self.afn_get_weight_and_bias_at_task(self.n_layers, stamp, task_id, "imp", is_bottom=True)
        y = tf.matmul(bottom, afn_w) + afn_b
        yhat = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))

        train_step = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)
        gradient_list = [tf.gradients(loss, h) for h in hidden_layer_list]

        self.sess.run(tf.global_variables_initializer())

        h_length_list = [h.get_shape().as_list()[-1] for h in hidden_layer_list]
        importance_vector_1 = np.zeros(shape=(0, h_length_list[0]))
        importance_vector_2 = np.zeros(shape=(0, h_length_list[1]))

        self.initialize_batch()
        while True:
            batch_x, batch_y = self.get_next_batch(self.trainXs[task_id - 1], self.mnist.train.labels)
            if len(batch_x) == 0:
                break

            # shape = (batch_size, |h|)
            if importance_criteria == "first_Taylor_approximation":
                batch_importance_vector_1, batch_importance_vector_2 = get_1st_taylor_approximation_based(self.sess, {
                    "hidden_layers": hidden_layer_list,
                    "gradients": gradient_list,
                }, {X: batch_x, Y: batch_y})

            elif importance_criteria == "activation":
                batch_importance_vector_1, batch_importance_vector_2 = get_activation_based(self.sess, {
                    "hidden_layers": hidden_layer_list,
                }, {X: batch_x, Y: batch_y})

            elif importance_criteria == "magnitude":
                batch_importance_vector_1, batch_importance_vector_2 = get_magnitude_based(self.sess, {
                    "weights": weight_list,
                    "biases": bias_list,
                }, {X: batch_x, Y: batch_y})

            elif importance_criteria == "gradient":
                batch_importance_vector_1, batch_importance_vector_2 = get_gradient_based(self.sess, {
                    "gradients": gradient_list,
                }, {X: batch_x, Y: batch_y})

            else:
                raise NotImplementedError

            importance_vector_1 = np.vstack((importance_vector_1, batch_importance_vector_1))
            importance_vector_2 = np.vstack((importance_vector_2, batch_importance_vector_2))

        importance_vector_1 = importance_vector_1.sum(axis=0)
        importance_vector_2 = importance_vector_2.sum(axis=0)

        if layer_separate:
            return importance_vector_1, importance_vector_2  # (|h1|,), (|h2|,)
        else:
            return np.concatenate((importance_vector_1, importance_vector_2))  # shape = (|h|,)

    # shape = (T, |h|) or (T, |h1|), (T, |h2|)
    def get_importance_matrix(self, layer_separate=False, importance_criteria=None):

        assert self.T > 0

        importance_matrix_1, importance_matrix_2 = None, None
        importance_criteria = importance_criteria or self.importance_criteria

        for t in reversed(range(1, self.T + 1)):
            iv_1, iv_2 = self.get_importance_vector(
                task_id=t,
                layer_separate=True,
                importance_criteria=importance_criteria,
            )

            if t == self.T:
                importance_matrix_1 = np.zeros(shape=(0, iv_1.shape[0]))
                importance_matrix_2 = np.zeros(shape=(0, iv_2.shape[0]))

            importance_matrix_1 = np.vstack((
                np.pad(iv_1, (0, importance_matrix_1.shape[-1] - iv_1.shape[0]), 'constant', constant_values=(0, 0)),
                importance_matrix_1,
            ))
            importance_matrix_2 = np.vstack((
                np.pad(iv_2, (0, importance_matrix_2.shape[-1] - iv_2.shape[0]), 'constant', constant_values=(0, 0)),
                importance_matrix_2,
            ))

        self.importance_matrix_tuple = importance_matrix_1, importance_matrix_2
        if layer_separate:
            return self.importance_matrix_tuple  # (T, |h1|), (T, |h2|)
        else:
            return np.concatenate(self.importance_matrix_tuple, axis=1)  # shape = (T, |h|)

    def get_neurons_by_mixed_ein_and_lin(self, task_id, number_to_select, sparsity_coeff=0.7, balanced=False):

        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        num_tasks, num_neurons = i_mat.shape

        ei = self.get_ei_value(task_id)
        minus_ei = - ei
        li = self.get_li_value(task_id, balanced)

        sparsity = number_to_select / num_neurons
        mixing_coeff = sparsity ** sparsity_coeff
        mixed = (1 - mixing_coeff) * (num_tasks - 1) * minus_ei + mixing_coeff * li

        mixed_asc_sorted_idx = np.argsort(mixed)
        selected = mixed_asc_sorted_idx[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def get_neurons_with_task_variance(self, task_id_or_ids, number_to_select, sparsity_coeff=0.2):

        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        if isinstance(task_id_or_ids, int):
            i_mat = np.delete(i_mat, task_id_or_ids - 1, axis=0)
        elif isinstance(task_id_or_ids, list):
            i_mat = np.delete(i_mat, [tid - 1 for tid in task_id_or_ids], axis=0)
        else:
            raise TypeError
        num_neurons = i_mat.shape[-1]

        li = self.get_li_value(task_id_or_ids)
        variance = np.std(i_mat, axis=0) ** 2

        sparsity = number_to_select / num_neurons
        mixing_coeff = sparsity ** sparsity_coeff
        mixed = (1 - mixing_coeff) * variance + mixing_coeff * li

        mixed_asc_sorted_idx = np.argsort(mixed)
        selected = mixed_asc_sorted_idx[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def get_ei_value(self, task_id):
        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        num_tasks, num_neurons = i_mat.shape

        mean_dot_j = np.mean(i_mat, axis=0)
        stdev_dot_j = np.std(i_mat, axis=0)

        ei = np.zeros(shape=(num_neurons,))
        for j in range(num_neurons):
            ei[j] = 1 / (num_tasks - 1) * (i_mat[task_id - 1][j] - mean_dot_j[j]) / (stdev_dot_j[j] + 1e-6)

        return ei

    def get_li_value(self, task_id_or_ids, balanced=False):
        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        if isinstance(task_id_or_ids, int):
            i_mat = np.delete(i_mat, task_id_or_ids - 1, axis=0)
        elif isinstance(task_id_or_ids, list):
            i_mat = np.delete(i_mat, [tid - 1 for tid in task_id_or_ids], axis=0)
        else:
            raise TypeError

        li = np.mean(i_mat, axis=0)

        if balanced:
            n_tasks_to_remove = 1 if isinstance(task_id_or_ids, int) else len(task_id_or_ids)
            n_tasks = self.n_tasks - n_tasks_to_remove

            time_stamps = [stamp for task_id, stamp in self.time_stamp.items()
                           if (isinstance(task_id_or_ids, int) and task_id != "task{}".format(task_id_or_ids)) or
                              (isinstance(task_id_or_ids, list) and task_id not in task_id_or_ids)]

            prev_layer1, prev_layer2 = 0, 0
            for (i, stamp) in enumerate(time_stamps):
                _, layer1, layer2, _ = stamp
                li[prev_layer1:layer1] *= n_tasks/(n_tasks - i)
                li[prev_layer2:layer2] *= n_tasks/(n_tasks - i)
                prev_layer1, prev_layer2 = layer1, layer2

        return li

    # Inappropriate for T=2
    def get_exceptionally_important_neurons_for_t(self, task_id, number_to_select):
        ei = self.get_ei_value(task_id)
        ei_desc_sorted_idx = np.argsort(ei)[::-1]
        selected = ei_desc_sorted_idx[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def get_least_important_neurons_for_others(self, task_id_or_ids: int or list, number_to_select, balanced=False):
        li = self.get_li_value(task_id_or_ids, balanced)
        li_asc_sorted_idx = np.argsort(li)
        selected = li_asc_sorted_idx[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def get_random_neurons(self, number_to_select):
        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        indexes = np.asarray(range(i_mat.shape[-1]))
        np.random.seed(i_mat.shape[-1])
        np.random.shuffle(indexes)
        selected = indexes[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def optimize_number_of_neurons(self, task_to_forget, policy, seed=42,
                                   init_points=5, n_iter=5, lmda=0.25, plot_optim_result=True):

        time_stamp = self.time_stamp["task{}".format(task_to_forget)]
        num_candidates_to_remove = sum(time_stamp[1:3])

        cprint("\n OPTIMIZE NUMBER OF NEURONS {} task-{} from {}".format(policy, task_to_forget, self.T), "green")

        # max f(n) = - err(n) + λ * sparsity(n)
        def get_perf_after_remove_n_neurons(x):

            n = int(x)

            self.adaptive_forget(task_to_forget, n, policy)
            preds = self.predict_only_after_training()
            errs = [1 - p for p in preds]

            # Mean error of tasks excl. the task to forget.
            mean_err_wo_task_to_forget = sum(e for i, e in enumerate(errs) if i != task_to_forget - 1)

            # sparsity = num of removed neurons / num of total neurons
            sparsity = n / num_candidates_to_remove

            self.recover_recent_params()

            return - mean_err_wo_task_to_forget + lmda * sparsity

        bounds = {'x': (1, num_candidates_to_remove)}

        bayes_optimizer = BayesianOptimization(
            f=get_perf_after_remove_n_neurons,
            pbounds=bounds,
            random_state=seed,
        )
        bayes_optimizer.maximize(init_points=init_points, n_iter=n_iter, kappa=5)

        if plot_optim_result:
            plot_gp(bayes_optimizer, bounds['x'], title="{} Bayes Opt. (λ = {})".format(policy, lmda))

        optimal_n = int(bayes_optimizer.max["params"]["x"])

        return optimal_n
