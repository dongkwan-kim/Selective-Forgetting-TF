from typing import List

import os
from pprint import pprint
import math

import tensorflow as tf
import numpy as np
from DEN.ops import ROC_AUC
from termcolor import cprint

from SFNBase import SFN
from enums import MaskType
from mask import Mask
from utils import get_dims_from_config, print_all_vars, with_tf_device_gpu, with_tf_device_cpu


class SFEWC(SFN):
    """
    Selective Forgettable Elastic Weight Consolidation

    Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks."
    Proceedings of the national academy of sciences. 2017.
    """

    def __init__(self, config):
        super(SFEWC, self).__init__(config)
        self.sess = tf.Session()
        self.batch_size = config.batch_size
        self.dims = get_dims_from_config(config)
        self.n_layers = len(self.dims) - 1
        self.n_classes = config.n_classes
        self.max_iter = config.max_iter
        self.init_lr = config.lr
        self.l1_lambda = config.l1_lambda
        self.l2_lambda = config.l2_lambda
        self.ewc_lambda = config.ewc_lambda
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.yhat = None
        self.loss = None
        self.loss_ewc = None
        self.fisher_matrices: List[np.ndarray] = []

        self.attr_to_save += ["fisher_matrices"]

        self.create_model_variables()
        self.set_layer_types()
        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")
        cprint("Device info: {}".format(self.get_real_device_info()), "green")

    def restore(self, model_name=None, model_middle_path=None, build_model=True):
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
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        test_preds = self.sess.run(self.yhat, feed_dict={X: xs})
        test_perf = self.get_performance(test_preds, ys)
        print(" [*] Evaluation, Task:%s, test perf: %.4f" % (str(task_id), test_perf))
        return test_perf

    def predict_only_after_training(self) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        temp_perfs = []
        for t in range(self.n_tasks):
            temp_perf = self.predict_perform(t + 1, self.testXs[t], self.data_labels.get_test_labels(t + 1))
            temp_perfs.append(temp_perf)
        print("   [*] avg_perf: %.4f +- %.4f" % (float(np.mean(temp_perfs)), float(np.std(temp_perfs))))
        return temp_perfs

    def set_layer_types(self, *args, **kwargs):
        for i in range(self.n_layers - 1):
            self.layer_types.append("layer")
        self.layer_types.append("layer")

    def create_model_variables(self):
        tf.reset_default_graph()
        for i in range(self.n_layers):
            w = self.create_variable('layer%d' % (i + 1), 'weight', [self.dims[i], self.dims[i + 1]])
            b = self.create_variable('layer%d' % (i + 1), 'biases', [self.dims[i + 1]])

    def build_model(self):

        X = tf.placeholder(tf.float32, [None, self.dims[0]], name="X")
        Y = tf.placeholder(tf.float32, [None, self.n_classes], name="Y")

        y = None
        bottom = X
        for i in range(1, self.n_layers + 1):
            w = self.get_variable('layer%d' % i, 'weight', True)
            b = self.get_variable('layer%d' % i, 'biases', True)

            if i < self.n_layers:
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            else:
                y = tf.matmul(bottom, w) + b

        self.yhat = tf.nn.softmax(y)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=Y))

        return X, Y

    def _get_fisher_matrix_loss(self, X, Y, task_id) -> tf.Variable:

        scope_list = self._get_scope_list()

        log_pred = tf.log(self.yhat)
        params_without_last_layer = [p for p in self.params.values() if not p.name.startswith(scope_list[-1])]

        # Construct fisher matrix
        gradient_list = []
        for i in range(self.n_classes):

            # list of Tensor the shape of which is (#params/layer,).
            gradients = [tf.reshape(g, [-1]) for g in tf.gradients(log_pred[:, i], params_without_last_layer)]

            # list of Tensor the shape of which is (total #params, 1).
            gradient_list.append(tf.expand_dims(tf.concat(gradients, axis=0), axis=1))

        gradient_tensor = tf.transpose(tf.concat(gradient_list, axis=1), [1, 0])  # (n_classes, total #params)

        fisher = tf.matmul(Y, gradient_tensor)  # (#samples, total #params)
        fisher_matrix = tf.reduce_mean(tf.multiply(fisher, fisher), axis=0)  # (total #params,)

        # Get value of FM
        xs, ys = self.trainXs[task_id - 1], self.data_labels.get_train_labels(task_id)
        random_permutation = np.random.permutation(len(xs))
        sampled_xs, sampled_ys = xs[random_permutation[:100]], ys[random_permutation[:100]]

        self.sess.run(tf.global_variables_initializer())
        fisher_matrix_value = self.sess.run(fisher_matrix, feed_dict={
            X: sampled_xs, Y: sampled_ys,
        })
        self.fisher_matrices.append(fisher_matrix_value)

        param_tf_vars = tf.concat([tf.reshape(p_var, [-1])
                                   for p_var in params_without_last_layer], axis=0)
        param_values = np.concatenate([np.reshape(p_val, [-1])
                                       for p_val in self.sess.run(params_without_last_layer)], axis=0)

        ewc_loss = None
        for fm_of_t in self.fisher_matrices:
            if ewc_loss is None:
                ewc_loss = tf.reduce_sum(tf.multiply(fm_of_t, tf.square(param_tf_vars - param_values)))
            else:
                ewc_loss += tf.reduce_sum(tf.multiply(fm_of_t, tf.square(param_tf_vars - param_values)))
        return ewc_loss

    @with_tf_device_gpu
    def initial_train(self, print_iter=100, *args):

        avg_perf = []

        self.sess = tf.Session()
        for t in range(self.n_tasks):

            X, Y = self.build_model()

            # Add L1 & L2 loss regularizer
            l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(
                scale_l1=self.l1_lambda,
                scale_l2=self.l2_lambda,
            )
            vars_of_task = []
            for var in tf.trainable_variables():
                if "layer" in var.name:
                    vars_of_task.append(var)
            regularization_loss = tf.contrib.layers.apply_regularization(l1_l2_regularizer, vars_of_task)
            self.loss += regularization_loss

            if t == 0:
                opt = tf.train.AdamOptimizer(learning_rate=self.init_lr,
                                             name="opt_{}".format(t + 1)).minimize(self.loss)

            else:
                self.loss_ewc = self.loss + self.ewc_lambda * self._get_fisher_matrix_loss(X, Y, t)
                opt = tf.train.AdamOptimizer(learning_rate=self.init_lr,
                                             name="opt_{}".format(t + 1)).minimize(self.loss_ewc)

            self.sess.run(tf.global_variables_initializer())
            self._initial_train_at_task(
                t=t, X=X, Y=Y, train_step=opt,
                loss=self.loss_ewc if t != 0 else self.loss,
                print_iter=print_iter,
            )

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
                params = self.get_params()
                self.clear()
                self.sess = tf.Session()
                self.load_params(params)

    def _initial_train_at_task(self, t, X, Y, train_step, loss, print_iter):
        train_xs_t, train_labels_t = self.trainXs[t], self.data_labels.get_train_labels(t + 1)
        val_xs_t, val_labels_t = self.valXs[t], self.data_labels.get_validation_labels(t + 1)
        for epoch in range(self.max_iter):
            self.initialize_batch()
            num_batches = int(math.ceil(len(train_xs_t) / self.batch_size))
            for _ in range(num_batches):
                batch_x, batch_y = self.get_next_batch(train_xs_t, train_labels_t)
                _, loss_val = self.sess.run([train_step, loss], feed_dict={X: batch_x, Y: batch_y})

            if epoch % print_iter == 0 or epoch == self.max_iter - 1:
                val_preds, val_loss_val = self.sess.run([self.yhat, loss], feed_dict={X: val_xs_t, Y: val_labels_t})
                val_perf = self.get_performance(val_preds, val_labels_t)
                print(" [*] iter: %d, val loss: %.4f, val perf: %.4f" % (epoch, val_loss_val, val_perf))

    @with_tf_device_gpu
    def get_importance_vector(self, task_id, importance_criteria: str,
                              layer_separate=False, use_coreset=False) -> tuple or np.ndarray:

        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y = tf.get_default_graph().get_tensor_by_name("Y:0")

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

        sfn_w = self.get_variable('layer%d' % self.n_layers, 'weight')
        sfn_b = self.get_variable('layer%d' % self.n_layers, 'biases')
        y = tf.matmul(bottom, sfn_w) + sfn_b
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=Y))
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
        self.loss_ewc = None
        self.yhat = None
        self.build_model()

    def build_model_for_retraining(self, flags):

        X_list = [tf.placeholder(tf.float32, [None, self.dims[0]], name="X_{}".format(t))
                  for t in range(self.n_tasks)]
        Y_list = [tf.placeholder(tf.float32, [None, self.n_classes], name="Y_{}".format(t))
                  for t in range(self.n_tasks)]
        loss_list = []

        for t, (X, Y) in enumerate(zip(X_list, Y_list)):

            if t + 1 in flags.task_to_forget:
                continue

            y = None
            bottom = X

            for i in range(1, self.n_layers + 1):
                w = self.get_variable('layer%d' % i, 'weight', True)
                b = self.get_variable('layer%d' % i, 'biases', True)

                if i < self.n_layers:
                    bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

                    ndarray_hard_mask = np.ones(shape=bottom.get_shape().as_list()[-1:])
                    indices_to_zero = np.asarray(list(self.layer_to_removed_unit_set["layer{}".format(i)]))
                    ndarray_hard_mask[indices_to_zero] = 0
                    tf_hard_mask = tf.constant(ndarray_hard_mask, dtype=tf.float32)

                    bottom = Mask(
                        i - 1, 0, 0, mask_type=MaskType.HARD, hard_mask=tf_hard_mask
                    ).get_masked_tensor(bottom)

                else:
                    y = tf.matmul(bottom, w) + b

            loss_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=Y)))

        l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.l1_lambda, scale_l2=self.l2_lambda)
        regularization_loss = tf.contrib.layers.apply_regularization(
            l1_l2_regularizer,
            [v for v in tf.trainable_variables() if "layer" in v.name]
        )

        loss = sum(loss_list) + regularization_loss
        opt = tf.train.AdamOptimizer(self.init_lr).minimize(loss)
        self.sess.run(tf.global_variables_initializer())

        return X_list, Y_list, opt, loss

    def _retrain_at_task_or_all(self, task_id, train_xs, train_labels, retrain_flags, is_verbose, **kwargs):
        X_list, Y_list, opt, loss = kwargs["model_args"]
        x_feed_dict = {X: train_xs[t]() for t, X in enumerate(X_list) if t + 1 not in retrain_flags.task_to_forget}
        y_feed_dict = {Y: train_labels[t]() for t, Y in enumerate(Y_list) if t + 1 not in retrain_flags.task_to_forget}
        _, loss_val = self.sess.run([opt, loss], feed_dict={**x_feed_dict, **y_feed_dict})
        return loss_val
