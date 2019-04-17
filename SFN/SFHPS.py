import os
from pprint import pprint

import tensorflow as tf
import numpy as np
from DEN.ops import ROC_AUC
from termcolor import cprint

from SFN.SFNBase import SFN
from utils import get_dims_from_config, print_all_vars


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
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.yhat_list = []
        self.loss_list = []

        self.create_model_variables()
        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")

    def restore(self, model_name=None):
        ret = super().restore(model_name)
        self.build_model()
        return ret

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

    def get_params(self):
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

    def get_performance(self, p, y):
        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:, _i], y[:, _i])  # TODO: remove DEN dependency
            perf_list.append(perf)
        return float(np.mean(perf_list))

    def predict_perform(self, task_id, xs, ys):
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        yhat = self.yhat_list[task_id - 1]
        test_preds = self.sess.run(yhat, feed_dict={X: xs})
        test_perf = self.get_performance(test_preds, ys)
        print(" [*] Evaluation, Task:%s, test perf: %.4f" % (str(task_id), test_perf))
        return test_perf

    def predict_only_after_training(self) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        temp_perfs = []
        for t in range(self.n_tasks):
            temp_perf = self.predict_perform(t + 1, self.testXs[t], self.mnist.test.labels)
            temp_perfs.append(temp_perf)
        return temp_perfs

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

            self.yhat_list.append(yhat)
            self.loss_list.append(loss)

        return X, Y_list

    def initial_train(self, print_iter=100):

        X, Y_list = self.build_model()

        # Add L2 loss regularizer
        for i in range(len(self.loss_list)):
            l2_losses = []
            for var in tf.trainable_variables():
                if "_" not in var.name or "_{}:".format(i+1) in var.name:
                    l2_losses.append(tf.nn.l2_loss(var))
            self.loss_list[i] += self.l2_lambda * tf.reduce_sum(l2_losses)

        opt_list = [tf.train.AdamOptimizer(learning_rate=self.init_lr, name="opt%d" % (i+1)).minimize(loss)
                    for i, loss in enumerate(self.loss_list)]

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
                self._train_at_task(X, Y_list[t], self.loss_list[t], opt_list[t], data)

            start_train_idx = next_train_idx % len(self.trainXs[0])

            if task_iter % print_iter == 0 or task_iter == self.task_iter - 1:
                print('\n OVERALL EVALUATION at ITERATION {}'.format(task_iter))
                overall_perfs = []
                for t in range(self.n_tasks):
                    temp_perf = self.predict_perform(t + 1, self.testXs[t], self.mnist.test.labels)
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
        _ = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)

        gradient_list = [tf.gradients(loss, h) for h in hidden_layer_list]
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

    def recover_params(self, idx):
        self.assign_new_session(idx)

    def assign_new_session(self, idx=None):
        if idx is None:
            params = self.get_params()
        else:
            params = self.old_params_list[idx]
        self.clear()
        self.sess = tf.Session()
        self.load_params(params)
        self.sess.run(tf.global_variables_initializer())
        self.loss_list = []
        self.yhat_list = []
        self.build_model()

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        pass

    def _assign_retrained_value_to_tensor(self, task_id):
        pass
