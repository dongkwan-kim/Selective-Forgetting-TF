import os
from pprint import pprint

import tensorflow as tf
import numpy as np
from DEN.ops import ROC_AUC

from SFN.SFNBase import SFN
from utils import get_dims_from_config, print_all_vars
from utils_importance import *


class SFHPS(SFN):
    """
    Selective Forgettable Hard Parameter Sharing MTL

    Caruana, R. "Multitask learning: A knowledge-based source of inductive bias."
    International Conference on Machine Learning. 1993.
    """

    def __init__(self, config):
        super(SFHPS, self).__init__(config)
        self.sess = tf.Session()
        self.batch_size = config.batch_size
        self.dims = get_dims_from_config(config)
        self.n_layers = len(self.dims) - 1
        self.n_classes = config.n_classes
        self.max_iter = config.max_iter
        self.task_iter = config.retrain_task_iter  # Just use retrain_task_ter for task_iter
        self.init_lr = config.lr
        self.l1_lambda = config.l1_lambda
        self.l2_lambda = config.l2_lambda
        self.checkpoint_dir = config.checkpoint_dir
        self.params = {}
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.create_model_variables()
        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")

    def create_variable(self, scope, name, shape, trainable=True):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable=trainable)
            self.params[w.name] = w
        return w

    def get_variable(self, scope, name, trainable=True):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(name, trainable=trainable)
            self.params[w.name] = w
        return w

    def get_performance(self, p, y):
        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:, _i], y[:, _i])
            perf_list.append(perf)
        return float(np.mean(perf_list))

    def predict_perform(self, task_id, xs, ys, X, yhat):
        test_preds = self.sess.run(yhat, feed_dict={X: xs})
        test_perf = self.get_performance(test_preds, ys)
        print(" [*] Evaluation, Task:%s, test perf: %.4f" % (str(task_id), test_perf))
        return test_perf

    def predict_only_after_training(self) -> list:
        pass

    def create_model_variables(self):
        tf.reset_default_graph()

        # Shared parameters
        for i in range(self.n_layers - 1):
            w = self.create_variable('layer%d' % (i + 1), 'weight', [self.dims[i], self.dims[i + 1]])
            b = self.create_variable('layer%d' % (i + 1), 'biases', [self.dims[i + 1]])

        # Task specific parameters
        for t in range(1, self.n_tasks + 1):
            w = self.create_variable('layer%d' % self.n_layers, 'weight_%d' % t, [self.dims[-2], self.dims[-1]])
            b = self.create_variable('layer%d' % self.n_layers, 'biases_%d' % t, [self.dims[-1]])

    def build_model(self):
        X = tf.placeholder(tf.float32, [None, self.dims[0]], name="X")
        Y_list = [tf.placeholder(tf.float32, [None, self.n_classes], name="Y_%d" % (t + 1))
                  for t in range(self.n_tasks)]
        yhat_list = []
        loss_list = []

        bottom = X
        for i in range(1, self.n_layers):
            w = self.get_variable('layer%d' % i, 'weight', True)
            b = self.get_variable('layer%d' % i, 'biases', True)
            bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

        for t in range(1, self.n_tasks + 1):
            w = self.get_variable("layer%d" % self.n_layers, "weight_%d" % t, True)
            b = self.get_variable("layer%d" % self.n_layers, "biases_%d" % t, True)
            y = tf.matmul(bottom, w) + b

            yhat = tf.nn.sigmoid(y)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y_list[t-1]))

            yhat_list.append(yhat)
            loss_list.append(loss)

        return X, Y_list, yhat_list, loss_list

    def initial_train(self, print_iter=100):

        X, Y_list, yhat_list, loss_list = self.build_model()

        # Add L2 loss regularizer
        for i in range(len(loss_list)):
            l2_losses = []
            for var in tf.trainable_variables():
                if "_" not in var.name or "_{}:".format(i+1) in var.name:
                    l2_losses.append(tf.nn.l2_loss(var))
            loss_list[i] += self.l2_lambda * tf.reduce_sum(l2_losses)

        opt_list = [tf.train.AdamOptimizer(learning_rate=self.init_lr, name="opt%d" % (i+1)).minimize(loss)
                    for i, loss in enumerate(loss_list)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        avg_perfs = []
        start_train_idx = 0
        for task_iter in range(self.task_iter):

            # Feed train data size of which is "self.max_iter * self.batch_size"
            next_train_idx = min(start_train_idx + self.max_iter * self.batch_size, len(self.trainXs[0]))

            for t in range(self.n_tasks):
                data = (self.trainXs[t][start_train_idx:next_train_idx],
                        self.mnist.train.labels[start_train_idx:next_train_idx],
                        self.valXs[t], self.mnist.validation.labels,
                        self.testXs[t], self.mnist.test.labels)
                self._train_at_task(X, Y_list[t], loss_list[t], opt_list[t], data)

            start_train_idx = next_train_idx % len(self.trainXs[0])

            if task_iter % print_iter == 0 or task_iter == self.task_iter - 1:
                print('\n OVERALL EVALUATION at ITERATION {}'.format(task_iter))
                overall_perfs = []
                for t in range(self.n_tasks):
                    temp_perf = self.predict_perform(t + 1, self.testXs[t], self.mnist.test.labels, X, yhat_list[t])
                    overall_perfs.append(temp_perf)
                avg_perfs.append(sum(overall_perfs) / float(self.n_tasks))
                print("   [*] avg_perf: %.4f" % avg_perfs[-1])

    def _train_at_task(self, X, Y, loss, train_step, data):
        train_xs_t, train_labels_t, val_xs_t, val_labels_t, test_xs_t, test_labels_t = data
        for epoch in range(self.max_iter):
            self.initialize_batch()
            while True:
                batch_x, batch_y = self.get_next_batch(train_xs_t, train_labels_t)
                if len(batch_x) == 0:
                    break
                _, loss_val = self.sess.run([train_step, loss], feed_dict={X: batch_x, Y: batch_y})

    # shape = (|h|,) or tuple of (|h1|,), (|h2|,)
    def get_importance_vector(self, task_id, importance_criteria: str, layer_separate=False) -> tuple or np.ndarray:
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y = tf.get_default_graph().get_tensor_by_name("Y_%d:0" % task_id)

        hidden_layer_list = []
        weight_list = []
        bias_list = []

        bottom = X
        for i in range(1, self.n_layers):
            sfn_w = self.get_variable('layer%d' % i, 'weight')
            sfn_b = self.get_variable('layer%d' % i, 'biases')
            bottom = tf.nn.relu(tf.matmul(bottom, sfn_w) + sfn_b)
            print(' [*] task %d, shape of layer %d : %s' % (task_id, i, sfn_w.get_shape().as_list()))

            hidden_layer_list.append(bottom)
            weight_list.append(sfn_w)
            bias_list.append(sfn_b)

        sfn_w = self.get_variable('layer%d' % self.n_layers, 'weight_%d' % task_id)
        sfn_b = self.get_variable('layer%d' % self.n_layers, 'biases_%d' % task_id)
        y = tf.matmul(bottom, sfn_w) + sfn_b
        yhat = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))
        train_step = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)

        gradient_list = [tf.gradients(loss, h) for h in hidden_layer_list]

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
            return np.concatenate((importance_vector_1, importance_vector_2))  # shape = (|h|,

    def recover_params(self, idx):
        pass

    def selective_forget(self, task_to_forget, number_of_neurons, policy) -> tuple:
        pass

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        pass

    def _assign_retrained_value_to_tensor(self, task_id):
        pass

    def assign_new_session(self):
        pass