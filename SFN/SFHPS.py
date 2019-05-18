from typing import Tuple, List

import os
from pprint import pprint
import math

import tensorflow as tf
import numpy as np
from DEN.ops import ROC_AUC
from termcolor import cprint
from tqdm import trange

from SFNBase import SFN
from utils import get_dims_from_config, print_all_vars, with_tf_device_cpu, with_tf_device_gpu

from cges.cges import cges


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
        self.init_lr = config.lr
        self.l1_lambda = config.l1_lambda
        self.l2_lambda = config.l2_lambda
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # CGES params
        self.use_cges = config.use_cges
        if self.use_cges:
            self.lr_decay_rate = config.lr_decay_rate
            self.cges_lambda = config.cges_lambda
            self.cges_mu = config.cges_mu
            self.cges_chvar = config.cges_chvar
            self.group_layerwise = [eval(str(gl)) for gl in config.group_layerwise]
            self.exclusive_layerwise = [eval(str(el)) for el in config.exclusive_layerwise]

        self.yhat_list = []
        self.loss = None

        self.create_model_variables()
        self.set_layer_types()
        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")
        cprint("Device info: {}".format(self.get_real_device_info()), "green")

    def save(self, model_name=None, model_middle_path=None):
        if self.use_cges:
            model_middle_path = "cges/"
        super().save(model_name=model_name, model_middle_path=model_middle_path)

    def restore(self, model_name=None, model_middle_path=None, build_model=True):
        if self.use_cges:
            model_middle_path = "cges/"
        restored = super().restore(model_name, model_middle_path, build_model)
        return restored

    @with_tf_device_cpu
    def create_variable(self, scope, name, shape, trainable=True) -> tf.Variable:
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable=trainable)
            self.params[w.name] = w
        return w

    @with_tf_device_cpu
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

    def get_performance(self, p, y):
        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:, _i], y[:, _i])  # TODO: remove DEN dependency
            perf_list.append(perf)
        return float(np.mean(perf_list))

    def predict_perform(self, task_id, xs, ys):
        X = tf.get_default_graph().get_tensor_by_name("X_{}:0".format(task_id))
        yhat = self.yhat_list[task_id - 1]
        test_preds = self.sess.run(yhat, feed_dict={X: xs})
        test_perf = self.get_performance(test_preds, ys)
        print(" [*] Evaluation, Task:%s, test perf: %.4f" % (str(task_id), test_perf))
        return test_perf

    def predict_only_after_training(self) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        temp_perfs = []
        for t in range(self.n_tasks):
            temp_perf = self.predict_perform(t + 1, self.testXs[t], self.data_labels.get_test_labels(t + 1))
            temp_perfs.append(temp_perf)
        return temp_perfs

    def evaluate_overall(self, iteration):
        cprint('\n OVERALL EVALUATION at ITERATION {} on Devices {}'.format(
            iteration, self.get_real_device_info()), "green")
        overall_perfs = []
        for t in range(self.n_tasks):
            temp_perf = self.predict_perform(t + 1, self.testXs[t], self.data_labels.get_test_labels(t + 1))
            overall_perfs.append(temp_perf)
        print("   [*] avg_perf: %.4f" % np.mean(overall_perfs))

    def set_layer_types(self):
        for i in range(self.n_layers - 1):
            self.layer_types.append("layer")
        self.layer_types.append("layer")

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

        X_list = [tf.placeholder(tf.float32, [None, self.dims[0]], name="X_%d" % (t + 1))
                  for t in range(self.n_tasks)]
        Y_list = [tf.placeholder(tf.float32, [None, self.n_classes], name="Y_%d" % (t + 1))
                  for t in range(self.n_tasks)]
        loss_list = []

        for t, (X, Y) in enumerate(zip(X_list, Y_list)):
            bottom = X
            for i in range(1, self.n_layers):
                w = self.get_variable('layer%d' % i, 'weight', True)
                b = self.get_variable('layer%d' % i, 'biases', True)
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w = self.get_variable("layer%d" % self.n_layers, "weight_%d" % (t + 1), True)
            b = self.get_variable("layer%d" % self.n_layers, "biases_%d" % (t + 1), True)
            y = tf.matmul(bottom, w) + b

            yhat = tf.nn.sigmoid(y)
            self.yhat_list.append(yhat)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))
            loss_list.append(loss)

        self.loss = sum(loss_list)

        return X_list, Y_list

    @with_tf_device_gpu
    def initial_train(self, print_iter=10):

        X_list, Y_list = self.build_model()

        # Add L1 & L2 loss regularizer
        l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.l1_lambda,
            scale_l2=self.l2_lambda,
        )
        variables = [var for var in tf.trainable_variables() if "weight" in var.name or "biases" in var.name]
        self.loss += tf.contrib.layers.apply_regularization(l1_l2_regularizer, variables)
        opt = tf.train.AdamOptimizer(learning_rate=self.init_lr, name="opt").minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        start_train_idx = 0
        batch_size_per_task = self.batch_size // self.n_tasks
        num_batches_per_task = int(math.ceil(len(self.trainXs[0]) / batch_size_per_task))
        for epoch in trange(self.max_iter):

            next_train_idx = min(start_train_idx + batch_size_per_task, len(self.trainXs[0]))

            for _ in range(num_batches_per_task):
                x_feed_dict = {X: self.trainXs[t][start_train_idx:next_train_idx] for t, X in enumerate(X_list)}
                y_feed_dict = {Y: self.data_labels.get_train_labels(t + 1)[start_train_idx:next_train_idx]
                               for t, Y in enumerate(Y_list)}

                _, loss_val = self.sess.run([opt, self.loss], feed_dict={**x_feed_dict, **y_feed_dict})

            start_train_idx = next_train_idx % len(self.trainXs[0])

            if epoch % print_iter == 0 or epoch == self.max_iter - 1:
                self.evaluate_overall(epoch)

    # shape = (|h|,) or tuple of (|h1|,), (|h2|,)
    @with_tf_device_gpu
    def get_importance_vector(self, task_id, importance_criteria: str,
                              layer_separate=False, use_coreset=False) -> tuple or np.ndarray:
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.get_default_graph().get_tensor_by_name("X_%d:0" % task_id)
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
            use_coreset=use_coreset,
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
        self.loss = None
        self.yhat_list = []
        self.build_model()

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        pass
