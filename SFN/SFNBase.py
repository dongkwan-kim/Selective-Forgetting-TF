from collections import defaultdict
from typing import Dict, List, Callable
import os
import pickle

import numpy as np
import tensorflow as tf
from termcolor import cprint

from data import PermutedCoreset
from utils import build_line_of_list, print_all_vars
from utils_importance import *


class SFN:

    def __init__(self, config):

        self.sess = None

        self.batch_size = config.batch_size
        self.checkpoint_dir = config.checkpoint_dir
        self.data_labels, self.trainXs, self.valXs, self.testXs = None, None, None, None
        self.n_tasks = config.n_tasks
        self.dims = None

        self.importance_matrix_tuple = None
        self.importance_criteria = config.importance_criteria

        self.prediction_history: Dict[str, List] = defaultdict(list)
        self.layer_to_removed_neuron_set: Dict[str, set] = defaultdict(set)
        self.batch_idx = 0
        self.retrained = False

        self.params = {}
        self.sfn_params = {}
        self.old_params_list = []

        self.attr_to_save = [
            "importance_matrix_tuple",
            "importance_criteria",
            "old_params_list",
            "layer_to_removed_neuron_set",
            "n_tasks",
        ]

    def __repr__(self):
        return "{}_{}_{}".format(self.__class__.__name__, self.n_tasks, "_".join(map(str, self.dims)))

    def add_dataset(self, data_labels, train_xs, val_xs, test_xs):
        self.data_labels, self.trainXs, self.valXs, self.testXs = data_labels, train_xs, val_xs, test_xs

    def predict_only_after_training(self) -> list:
        raise NotImplementedError

    def initial_train(self, *args):
        raise NotImplementedError

    # Variable, params, ... attributes Manipulation

    def get_params(self) -> dict:
        raise NotImplementedError

    def load_params(self, params, *args, **kwargs):
        raise NotImplementedError

    def recover_params(self, idx):
        raise NotImplementedError

    def create_variable(self, scope, name, shape, trainable=True) -> tf.Variable:
        raise NotImplementedError

    def get_variable(self, scope, name, trainable=True) -> tf.Variable:
        raise NotImplementedError

    def assign_new_session(self, idx_to_load_params=None):
        """
        :param idx_to_load_params: if idx_to_load_params is None, use current params
                                   else use self.old_params_list[idx_to_load_params]
        :return:
        """
        raise NotImplementedError

    def clear(self):
        tf.reset_default_graph()
        self.sess.close()

    def sfn_create_variable(self, scope, name, shape=None, trainable=True, initializer=None):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
            if 'new' not in w.name:
                self.sfn_params[w.name] = w
        return w

    def sfn_get_variable(self, scope, name, trainable=True):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(name, trainable=trainable)
            self.sfn_params[w.name] = w
        return w

    def sfn_create_or_get_variable(self, scope, name, shape=None, trainable=True, initializer=None):
        try:
            w = self.sfn_create_variable(scope, name, shape, trainable, initializer)
        except ValueError:
            w = self.sfn_get_variable(scope, name, trainable)
        return w

    def sfn_get_params(self, name_filter: Callable = None):
        """ Access the sfn_parameters """
        mdict = dict()
        for scope_name, param in self.sfn_params.items():
            if name_filter is None or name_filter(scope_name):
                w = self.sess.run(param)
                mdict[scope_name] = w
        return mdict

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

    def restore(self, model_name=None) -> bool:
        model_name = model_name or str(self)
        model_path = os.path.join(self.checkpoint_dir, "{}.ckpt".format(model_name))

        if not os.path.isfile("{}.meta".format(model_path)):
            return False

        try:
            # Attribute Restore
            self.restore_attr(model_name)

            # Recreate variables
            self.create_model_variables()

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

    def create_model_variables(self):
        raise NotImplementedError

    # Data batch

    def initialize_batch(self):
        self.batch_idx = 0

    def get_next_batch(self, x, y, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size
        next_idx = self.batch_idx + batch_size
        r = x[self.batch_idx:next_idx], y[self.batch_idx:next_idx]
        self.batch_idx = next_idx
        return r

    # Data visualization

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
        min_acc_except_t = None
        x_removed_neurons = None

        for policy, history in self.prediction_history.items():

            x_removed_neurons = [i * one_step_neuron for i, acc in enumerate(history)]
            history_txn = np.transpose(history)
            tasks = [x for x in range(1, self.n_tasks + 1)]

            build_line_of_list(x=x_removed_neurons, y_list=history_txn, label_y_list=tasks,
                               xlabel="Removed Neurons", ylabel="Accuracy",
                               ylim=ylim or [0.5, 1],
                               title="Accuracy by {} Neuron Deletion".format(policy),
                               file_name="{}_{}_{}{}".format(
                                   file_prefix, self.importance_criteria.split("_")[0], policy, file_extension),
                               highlight_yi=task_id - 1)

            history_txn_except_t = np.delete(history_txn, task_id - 1, axis=0)
            history_n_mean_except_t = np.mean(history_txn_except_t, axis=0)
            history_n_min_except_t = np.min(history_txn_except_t, axis=0)

            if mean_acc_except_t is None:
                mean_acc_except_t = history_n_mean_except_t
                min_acc_except_t = history_n_min_except_t
            else:
                mean_acc_except_t = np.vstack((mean_acc_except_t, history_n_mean_except_t))
                min_acc_except_t = np.vstack((min_acc_except_t, history_n_min_except_t))

        build_line_of_list(x=x_removed_neurons, y_list=mean_acc_except_t,
                           label_y_list=[policy for policy in self.prediction_history.keys()],
                           xlabel="Removed Neurons", ylabel="Mean Accuracy",
                           ylim=ylim or [0.7, 1],
                           title="Mean Accuracy Except Forgetting Task-{}".format(task_id),
                           file_name="{}_{}_MeanAcc{}".format(
                               file_prefix, self.importance_criteria.split("_")[0], file_extension,
                           ))
        build_line_of_list(x=x_removed_neurons, y_list=min_acc_except_t,
                           label_y_list=[policy for policy in self.prediction_history.keys()],
                           xlabel="Removed Neurons", ylabel="Min Accuracy",
                           ylim=ylim or [0.5, 1],
                           title="Minimum Accuracy Except Forgetting Task-{}".format(task_id),
                           file_name="{}_{}_MinAcc{}".format(
                               file_prefix, self.importance_criteria.split("_")[0], file_extension,
                           ))

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

    # Pruning strategies

    def _get_reduced_i_mat(self, task_id_or_ids):
        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        if isinstance(task_id_or_ids, int):
            i_mat = np.delete(i_mat, task_id_or_ids - 1, axis=0)
        elif isinstance(task_id_or_ids, list):
            i_mat = np.delete(i_mat, [tid - 1 for tid in task_id_or_ids], axis=0)
        else:
            raise TypeError
        return i_mat

    def get_exceptionally_importance(self, task_id):
        # TODO: not only task_id (int) but also task_ids (list)
        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        num_tasks, num_neurons = i_mat.shape

        mean_dot_j = np.mean(i_mat, axis=0)
        stdev_dot_j = np.std(i_mat, axis=0)

        ei = np.zeros(shape=(num_neurons,))
        for j in range(num_neurons):
            ei[j] = 1 / (num_tasks - 1) * (i_mat[task_id - 1][j] - mean_dot_j[j]) / (stdev_dot_j[j] + 1e-6)

        return ei

    def get_least_importance(self, task_id_or_ids):
        i_mat = self._get_reduced_i_mat(task_id_or_ids)
        li = np.mean(i_mat, axis=0)
        return li

    def get_maximum_importance(self, task_id_or_ids):
        i_mat = self._get_reduced_i_mat(task_id_or_ids)
        return np.max(i_mat, axis=0)

    def get_neurons_by_mixed_ein_and_lin(self, task_id, number_to_select, mixing_coeff=0.45):

        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        num_tasks, num_neurons = i_mat.shape

        li = self.get_least_importance(task_id)

        if isinstance(task_id, int) or (isinstance(task_id, list) and len(task_id) != 0):
            ei = self.get_exceptionally_importance(task_id)
            minus_ei = - ei
        else:  # EI cannot be calculated while forgetting zero task.
            minus_ei = 0

        sparsity = number_to_select / num_neurons
        mixed = (1 - mixing_coeff) * (num_tasks - 1) * minus_ei + mixing_coeff * li

        mixed_asc_sorted_idx = np.argsort(mixed)
        selected = mixed_asc_sorted_idx[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def get_neurons_with_task_variance(self, task_id_or_ids, number_to_select, mixing_coeff=0.65):

        i_mat = self._get_reduced_i_mat(task_id_or_ids)
        num_tasks, num_neurons = i_mat.shape

        li = self.get_least_importance(task_id_or_ids)
        variance = np.std(i_mat, axis=0) ** 2

        sparsity = number_to_select / num_neurons
        mixed = (1 - mixing_coeff) * variance + mixing_coeff * li

        mixed_asc_sorted_idx = np.argsort(mixed)
        selected = mixed_asc_sorted_idx[:number_to_select]
        divider = self.importance_matrix_tuple[0].shape[-1]
        return selected[selected < divider], (selected[selected >= divider] - divider)

    def get_neurons_with_maximum_importance(self, task_id_or_ids, number_to_select, mixing_coeff=0.65):

        i_mat = np.concatenate(self.importance_matrix_tuple, axis=1)
        num_tasks, num_neurons = i_mat.shape

        li = self.get_least_importance(task_id_or_ids)
        mi = self.get_maximum_importance(task_id_or_ids)

        sparsity = number_to_select / num_neurons
        mixed = (1 - mixing_coeff) * mi + mixing_coeff * li

        mi_asc_sorted_idx = np.argsort(mixed)
        selected = mi_asc_sorted_idx[:number_to_select]
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

    # Selective forgetting

    def selective_forget(self, task_to_forget, number_of_neurons, policy, **kwargs) -> tuple:

        assert policy in ["MIX", "MAX", "VAR", "EIN", "LIN", "RANDOM", "ALL", "ALL_VAR"]

        self.old_params_list.append(self.get_params())

        if not self.importance_matrix_tuple:
            self.get_importance_matrix()

        cprint("\n SELECTIVE FORGET {} task-{} from {}, neurons-{}".format(
            policy, task_to_forget, self.n_tasks, number_of_neurons), "green")

        if policy == "MIX":
            neuron_indexes = self.get_neurons_by_mixed_ein_and_lin(task_to_forget, number_of_neurons, **kwargs)
        elif policy == "MAX":
            neuron_indexes = self.get_neurons_with_maximum_importance(task_to_forget, number_of_neurons, **kwargs)
        elif policy == "VAR":
            neuron_indexes = self.get_neurons_with_task_variance(task_to_forget, number_of_neurons, **kwargs)
        elif policy == "EIN":
            neuron_indexes = self.get_neurons_by_mixed_ein_and_lin(task_to_forget, number_of_neurons, mixing_coeff=0)
        elif policy == "LIN":
            neuron_indexes = self.get_neurons_by_mixed_ein_and_lin(task_to_forget, number_of_neurons, mixing_coeff=1)
        elif policy == "RANDOM":
            neuron_indexes = self.get_random_neurons(number_of_neurons)
        elif policy == "ALL":
            neuron_indexes = self.get_neurons_by_mixed_ein_and_lin([], number_of_neurons, mixing_coeff=1)
        elif policy == "ALL_VAR":
            neuron_indexes = self.get_neurons_with_task_variance([], number_of_neurons, **kwargs)
        else:
            raise NotImplementedError

        for i, ni in enumerate(neuron_indexes):
            self._remove_neurons("layer{}".format(i + 1), ni)

        self.assign_new_session()

        return neuron_indexes

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

    def sequentially_selective_forget_and_predict(self, task_to_forget, one_step_neurons, steps, policy):

        cprint("\n SEQUENTIALLY SELECTIVE FORGET {} task-{} from {}, neurons-{}".format(
            policy, task_to_forget, self.n_tasks, one_step_neurons * steps), "green")

        for i in range(steps + 1):
            self.selective_forget(task_to_forget, i * one_step_neurons, policy)
            pred = self.predict_only_after_training()
            self.prediction_history[policy].append(pred)

            if i != steps:
                self.recover_recent_params()

    # Importance vectors
    def get_importance_vector(self, task_id, importance_criteria: str, layer_separate=False) -> tuple or np.ndarray:
        """
        :param task_id:
        :param importance_criteria:
        :param layer_separate:
            - layer_separate = True: tuple of ndarray of shape (|h1|,), (|h2|,) or
            - layer_separate = False: ndarray of shape (|h|,)

        First construct the model, then pass tf vars to get_importance_vector_from_tf_vars

        :return: The return value of get_importance_vector_from_tf_vars
        """
        raise NotImplementedError

    def get_importance_vector_from_tf_vars(self, task_id, importance_criteria,
                                           h_length_list, hidden_layer_list, gradient_list, weight_list, bias_list,
                                           X, Y, layer_separate=False) -> tuple or np.ndarray:

        importance_vectors = [np.zeros(shape=(0, h_length)) for h_length in h_length_list]

        self.initialize_batch()
        while True:
            batch_x, batch_y = self.get_next_batch(self.trainXs[task_id - 1], self.data_labels.train_labels)
            if len(batch_x) == 0:
                break

            # shape = (batch_size, |h|)
            if importance_criteria == "first_Taylor_approximation":
                batch_importance_vectors = get_1st_taylor_approximation_based(self.sess, {
                    "hidden_layers": hidden_layer_list,
                    "gradients": gradient_list,
                }, {X: batch_x, Y: batch_y})

            elif importance_criteria == "activation":
                batch_importance_vectors = get_activation_based(self.sess, {
                    "hidden_layers": hidden_layer_list,
                }, {X: batch_x, Y: batch_y})

            elif importance_criteria == "magnitude":
                batch_importance_vectors = get_magnitude_based(self.sess, {
                    "weights": weight_list,
                    "biases": bias_list,
                }, {X: batch_x, Y: batch_y})

            elif importance_criteria == "gradient":
                batch_importance_vectors = get_gradient_based(self.sess, {
                    "gradients": gradient_list,
                }, {X: batch_x, Y: batch_y})

            else:
                raise NotImplementedError

            for i, batch_i_vector in enumerate(batch_importance_vectors):
                importance_vectors[i] = np.vstack((importance_vectors[i], batch_i_vector))

        for i in range(len(importance_vectors)):
            importance_vectors[i] = importance_vectors[i].sum(axis=0)

        if layer_separate:
            return tuple(importance_vectors)  # (|h1|,), (|h2|,)
        else:
            return np.concatenate(importance_vectors)  # shape = (|h|,)

    # shape = (T, |h|) or (T, |h1|), (T, |h2|)
    def get_importance_matrix(self, layer_separate=False, importance_criteria=None):

        # TODO: handle more than two layer networks
        importance_matrices = []

        importance_criteria = importance_criteria or self.importance_criteria
        self.importance_criteria = importance_criteria

        for t in reversed(range(1, self.n_tasks + 1)):
            i_vector_tuple = self.get_importance_vector(
                task_id=t,
                layer_separate=True,
                importance_criteria=importance_criteria,
            )

            if t == self.n_tasks:
                for iv in i_vector_tuple:
                    importance_matrices.append(np.zeros(shape=(0, iv.shape[0])))

            for i, iv in enumerate(i_vector_tuple):
                imat = importance_matrices[i]
                importance_matrices[i] = np.vstack((
                    np.pad(iv, (0, imat.shape[-1] - iv.shape[0]), 'constant', constant_values=(0, 0)),
                    imat,
                ))

        self.importance_matrix_tuple = tuple(importance_matrices)
        if layer_separate:
            return self.importance_matrix_tuple  # (T, |h1|), (T, |h2|)
        else:
            return np.concatenate(self.importance_matrix_tuple, axis=1)  # shape = (T, |h|)

    # Retrain after forgetting

    def retrain_after_forgetting(self, flags, policy, coreset: PermutedCoreset = None,
                                 epoches_to_print: list = None, is_verbose: bool = True):
        cprint("\n RETRAIN AFTER FORGETTING - {}".format(policy), "green")
        self.retrained = True
        series_of_perfs = []

        # First, append perfs wo/ retraining
        perfs = self.predict_only_after_training()
        series_of_perfs.append(perfs)

        for retrain_iter in range(flags.retrain_task_iter):

            cprint("\n\n\tRE-TRAINING at iteration %d\n" % retrain_iter, "green")

            for t in range(flags.n_tasks):
                if (t + 1) != flags.task_to_forget:
                    coreset_t = coreset[t] if coreset is not None \
                                           else (self.trainXs[t], self.data_labels.train_labels,
                                                 self.valXs[t], self.data_labels.validation_labels,
                                                 self.testXs[t], self.data_labels.test_labels)
                    if is_verbose:
                        cprint("\n\n\tTASK %d RE-TRAINING at iteration %d\n" % (t + 1, retrain_iter), "green")
                    self._retrain_at_task(t + 1, coreset_t, flags, is_verbose)
                    self._assign_retrained_value_to_tensor(t + 1)
                    self.assign_new_session()
                else:
                    if is_verbose:
                        cprint("\n\n\tTASK %d NO NEED TO RE-TRAIN at iteration %d" % (t + 1, retrain_iter), "green")

            perfs = self.predict_only_after_training()
            series_of_perfs.append(perfs)

        if epoches_to_print:
            print("\t".join(str(t + 1) for t in range(self.n_tasks)))
            for epo in epoches_to_print:
                print("\t".join(str(round(acc, 4)) for acc in series_of_perfs[epo]))

        return series_of_perfs

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        raise NotImplementedError

    def _assign_retrained_value_to_tensor(self, task_id):
        raise NotImplementedError
