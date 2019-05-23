import os
import re
from pprint import pprint
from typing import Tuple, List
import math

import tensorflow as tf
import numpy as np
from DEN.ops import ROC_AUC, accuracy
from termcolor import cprint
from tqdm import trange

from SFNBase import SFN
from enums import MaskType
from mask import Mask
from utils import get_dims_from_config, print_all_vars, \
    with_tf_device_gpu, with_tf_device_cpu, get_middle_path_name, cprint_stats_of_mask_pair, get_batch_iterator


def parse_pool_key(pool_name):
    p = re.compile("pool(\\d+)_ksize")
    return int(p.findall(pool_name)[0])


def _get_batch_normalized_conv(beta: tf.Variable,
                               gamma: tf.Variable,
                               h_conv: tf.Tensor,
                               is_training: tf.Variable) -> tf.Tensor:
    """Ref. https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412"""

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_variance])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_variance)

    batch_mean, batch_variance = tf.nn.moments(h_conv, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, variance = tf.cond(is_training,
                             mean_var_with_update,
                             lambda: (ema.average(batch_mean), ema.average(batch_variance)))
    h_conv = tf.nn.batch_normalization(h_conv, mean, variance, beta, gamma, 1e-3)
    return h_conv


def _get_xs_and_labels_wo_target(target_t, train_xs, train_labels):
    xs_except_target = np.concatenate([xs() for i, xs in enumerate(train_xs) if i != target_t])
    labels_except_target = np.concatenate([label() for i, label in enumerate(train_labels) if i != target_t])
    permut = np.random.permutation(len(xs_except_target))
    xs_except_target, labels_except_target = xs_except_target[permut], labels_except_target[permut]
    return xs_except_target, labels_except_target


class SFLCL(SFN):
    """
    Selective Forgettable AlexNet Variant for Large Class Learning

    Krizhevsky et al. "ImageNet Classification with Deep Convolutional Neural Networks"
    Advances in Neural Information Processing Systems 2015.

    ref. https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
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
        self.dropout_type = config.dropout_type if "dropout_type" in config else "dropout"
        self.keep_prob = config.keep_prob if "keep_prob" in config else 1
        self.use_batch_normalization = config.use_batch_normalization
        assert not self.use_batch_normalization, "Not support batch_norm, yet"

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

        self.use_set_based_mask = config.use_set_based_mask
        if self.use_set_based_mask:
            self.mask_type = config.mask_type
            self.mask_alpha = config.mask_alpha if isinstance(config.mask_alpha, list) \
                else [config.mask_alpha for _ in range(len(self.conv_dims))]
            self.mask_not_alpha = config.mask_not_alpha if isinstance(config.mask_not_alpha, list) \
                else [config.mask_not_alpha for _ in range(len(self.conv_dims))]
            self.excl_loss = None

        self.yhat = None
        self.loss = None
        self.validation_results = []

        self.create_model_variables()
        self.set_layer_types()
        self.attr_to_save += ["max_iter", "l1_lambda", "l2_lambda", "keep_prob", "gpu_names", "use_batch_normalization"]
        print_all_vars("{} initialized:".format(self.__class__.__name__), "green")
        cprint("Device info: {}".format(self.get_real_device_info()), "green")

    def save(self, model_name=None, model_middle_path=None):
        model_middle_path = get_middle_path_name({
            "sbm": self.use_set_based_mask,
            "cges": self.use_cges,
            **{d: True for d in self.get_real_device_info()}
        })
        super().save(model_name=model_name, model_middle_path=model_middle_path)

    def restore(self, model_name=None, model_middle_path=None, build_model=True):
        model_middle_path = get_middle_path_name({
            "sbm": self.use_set_based_mask,
            "cges": self.use_cges,
            **{d: True for d in self.get_real_device_info()}
        })
        restored = super().restore(model_name, model_middle_path, build_model)
        return restored

    @with_tf_device_cpu
    def create_variable(self, scope, name, shape, trainable=True, **kwargs) -> tf.Variable:
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable=trainable, **kwargs)
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

    def get_performance(self, p, y) -> list:
        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:, _i], y[:, _i])  # TODO: remove DEN dependency
            perf_list.append(perf)
        return [float(p) for p in perf_list]

    def predict_perform(self, xs, ys, number_to_print=8, with_residuals=False) -> list or tuple:
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
        test_preds_list = []

        # (10000, 32, 32, 3) is too big, so divide to batches.
        test_batch_size = len(xs) // 5
        for i in range(5):
            partial_xs = xs[i * test_batch_size:(i + 1) * test_batch_size]
            test_preds_list.append(self.sess.run(self.yhat,
                                                 feed_dict={X: partial_xs, keep_prob: 1, is_training: False}))
        test_preds = np.concatenate(test_preds_list)

        half_number_to_print = int(number_to_print / 2)
        print(" [*] Evaluation, ")
        test_perf = self.get_performance(test_preds, ys)
        for rank, idx in enumerate(reversed(np.argsort(test_perf))):
            if rank < half_number_to_print:
                print("\t Class of task_id: %s, test perf: %.4f" % (str(idx + 1), test_perf[idx]))
            elif rank >= len(test_perf) - half_number_to_print:
                print("\t Class of task_id: %s, test perf: %.4f" % (str(idx + 1), test_perf[idx]))

            if rank == half_number_to_print and len(test_perf) > 2 * half_number_to_print:
                print("\t ...")

        accuracy_value = accuracy(test_preds, ys)
        print("   [*] avg_perf: %.4f +- %.4f" % (float(np.mean(test_perf)), float(np.std(test_perf))))
        print("   [*] accuracy: {} %".format(accuracy_value))
        if with_residuals:
            return test_perf, {
                "accuracy": accuracy_value,
            }
        else:
            return test_perf

    def predict_only_after_training(self, **kwargs) -> list:
        cprint("\n PREDICT ONLY AFTER " + ("TRAINING" if not self.retrained else "RE-TRAINING"), "yellow")
        _, _, _, _, test_x, test_labels = self._get_data_stream_from_task_as_class_data(shuffle=False)
        perfs = self.predict_perform(test_x, test_labels, **kwargs)
        return perfs

    def evaluate_overall(self, iteration, val_x, val_labels, loss_sum):
        cprint('\n OVERALL EVALUATION at ITERATION {} on Devices {}'.format(
            iteration, self.get_real_device_info()), "green")
        validation_perf, residuals = self.predict_perform(val_x, val_labels, with_residuals=True)
        self.validation_results.append((np.mean(validation_perf), residuals["accuracy"]))
        print("   [*] max avg_perf: %.4f" % max(apf for apf, acc in self.validation_results))
        print("   [*] max accuracy: {} %".format(max(acc for apf, acc in self.validation_results)))
        print("   [*] training loss: {}".format(loss_sum))

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

            if self.use_batch_normalization:
                beta = self.create_variable("bn%d" % (i // 2), "beta", (n_filters,),
                                            initializer=tf.constant_initializer(0.0))
                gamma = self.create_variable("bn%d" % (i // 2), "gamma", (n_filters,),
                                             initializer=tf.constant_initializer(1.0))

        for i in range(1, len(self.dims)):
            prev_dims, curr_dims = self.dims[i - 1], self.dims[i]
            w_fc = self.create_variable("fc%d" % i, "weight", (prev_dims, curr_dims))
            b_fc = self.create_variable("fc%d" % i, "biases", (curr_dims,))

    def build_model(self):

        xn_filters, xsize = self.conv_dims[0], self.conv_dims[1]
        X = tf.placeholder(tf.float32, [None, xsize, xsize, xn_filters], name="X")
        Y = tf.placeholder(tf.float32, [None, self.n_classes], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        is_training = tf.placeholder(tf.bool, name="is_training")

        excl_X = tf.placeholder(tf.float32, [None, xsize, xsize, xn_filters], name="excl_X")
        excl_Y = tf.placeholder(tf.float32, [None, self.n_classes], name="excl_Y")

        h_conv = X
        excl_h_conv = excl_X if self.use_set_based_mask else None
        for conv_num, i in enumerate(range(2, len(self.conv_dims), 2)):
            w_conv = self.get_variable("conv%d" % (i // 2), "weight", True)
            b_conv = self.get_variable("conv%d" % (i // 2), "biases", True)
            h_conv = tf.nn.conv2d(h_conv, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv
            if self.use_set_based_mask:
                excl_h_conv = tf.nn.conv2d(excl_h_conv, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv

            # batch normalization
            if self.use_batch_normalization:
                beta = self.get_variable("bn%d" % (i // 2), "beta", True)
                gamma = self.get_variable("bn%d" % (i // 2), "gamma", True)
                h_conv = _get_batch_normalized_conv(beta, gamma, h_conv, is_training)
                if self.use_set_based_mask:
                    excl_h_conv = _get_batch_normalized_conv(beta, gamma, excl_h_conv, is_training)

            # activation
            h_conv = tf.nn.relu(h_conv)

            # Mask
            if self.use_set_based_mask:
                h_conv, residuals = Mask(
                    conv_num, self.mask_alpha[conv_num], self.mask_not_alpha[conv_num], mask_type=self.mask_type,
                ).get_masked_tensor(h_conv, is_training, with_residuals=True)
                mask = residuals["cond_mask"]
                with tf.control_dependencies([mask]):
                    excl_h_conv = tf.nn.relu(excl_h_conv)
                    excl_h_conv = Mask.get_exclusive_masked_tensor(excl_h_conv, mask, is_training)

            # max pooling
            if conv_num + 1 in self.pool_pos_to_dims:
                pool_dim = self.pool_pos_to_dims[conv_num + 1]
                pool_ksize = [1, pool_dim, pool_dim, 1]
                h_conv = tf.nn.max_pool(h_conv, ksize=pool_ksize, strides=[1, 2, 2, 1], padding="SAME")
                if self.use_set_based_mask:
                    excl_h_conv = tf.nn.max_pool(excl_h_conv, ksize=pool_ksize, strides=[1, 2, 2, 1], padding="SAME")

        h_fc = tf.reshape(h_conv, (-1, self.dims[0]))
        excl_h_fc = tf.reshape(excl_h_conv, (-1, self.dims[0])) if self.use_set_based_mask else None
        for i in range(1, len(self.dims)):
            w_fc = self.get_variable("fc%d" % i, "weight", True)
            b_fc = self.get_variable("fc%d" % i, "biases", True)
            h_fc = tf.matmul(h_fc, w_fc) + b_fc
            if self.use_set_based_mask:
                excl_h_fc = tf.matmul(excl_h_fc, w_fc) + b_fc

            if i < len(self.dims) - 1:  # Do not activate the last layer.
                h_fc = tf.nn.relu(h_fc)
                if self.use_set_based_mask:
                    excl_h_fc = tf.nn.relu(excl_h_fc)

                if self.dropout_type == "dropout":
                    h_fc = tf.nn.dropout(h_fc, keep_prob)
                    if self.use_set_based_mask:
                        excl_h_fc = tf.nn.dropout(excl_h_fc, keep_prob)
                else:
                    raise ValueError

        self.yhat = tf.nn.sigmoid(h_fc)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h_fc, labels=Y))
        if self.use_set_based_mask:
            self.excl_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=excl_h_fc, labels=excl_Y))

        return X, Y, excl_X, excl_Y, keep_prob, is_training

    @with_tf_device_gpu
    def initial_train(self, print_iter=5, *args):

        X, Y, excl_X, excl_Y, keep_prob, is_training = self.build_model()

        # Add L1 & L2 loss regularizer
        l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.l1_lambda,
            scale_l2=self.l2_lambda,
        )
        variables = [var for var in tf.trainable_variables() if "conv" in var.name or "fc" in var.name]
        regularization_loss = tf.contrib.layers.apply_regularization(l1_l2_regularizer, variables)
        train_x, train_labels, val_x, val_labels, test_x, test_labels = self._get_data_stream_from_task_as_class_data()
        num_batches = int(math.ceil(len(train_x) / self.batch_size))

        if not self.use_set_based_mask:
            loss = self.loss + regularization_loss
            opt = tf.train.AdamOptimizer(learning_rate=self.init_lr, name="opt").minimize(loss)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            for epoch in trange(self.max_iter):
                self.initialize_batch()
                loss_sum = 0
                for _ in range(num_batches):
                    batch_x, batch_y = self.get_next_batch(train_x, train_labels)
                    _, loss_val = self.sess.run(
                        [opt, loss],
                        feed_dict={X: batch_x, Y: batch_y, keep_prob: self.keep_prob, is_training: True},
                    )
                    loss_sum += loss_val

                if epoch % print_iter == 0 or epoch == self.max_iter - 1:
                    self.evaluate_overall(epoch, val_x, val_labels, loss_sum)
        else:
            loss_with_excl = self.loss + self.excl_loss + regularization_loss
            opt_with_excl = tf.train.AdamOptimizer(learning_rate=self.init_lr, name="opt").minimize(loss_with_excl)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            batch_size_per_task = self.batch_size // self.n_tasks

            xs_queues = [get_batch_iterator(self.trainXs[t], batch_size_per_task) for t in range(self.n_tasks)]
            labels_queues = [get_batch_iterator(self.data_labels.get_train_labels(t + 1), batch_size_per_task)
                             for t in range(self.n_tasks)]

            target_t = 0
            for epoch in trange(self.max_iter):
                loss_sum = 0
                for _ in range(num_batches):
                    xs_wo_target, labels_wo_target = _get_xs_and_labels_wo_target(target_t, xs_queues, labels_queues)

                    feed_dict = {
                        X: xs_queues[target_t](), Y: labels_queues[target_t](),
                        excl_X: xs_wo_target, excl_Y: labels_wo_target,
                        keep_prob: self.keep_prob, is_training: True,
                    }
                    _, loss_val = self.sess.run([opt_with_excl, loss_with_excl], feed_dict=feed_dict)
                    loss_sum += loss_val
                    target_t = (target_t + 1) % self.n_tasks

                if epoch % print_iter == 0 or epoch == self.max_iter - 1:
                    self.evaluate_overall(epoch, val_x, val_labels, loss_sum)
                    for i in range(len(self.conv_dims) // 2 - 1):
                        cprint_stats_of_mask_pair(self, 1, 6, batch_size_per_task, X, is_training, mask_id=i)

    def _get_data_stream_from_task_as_class_data(self, shuffle=True, base_seed=42) -> Tuple[np.ndarray, ...]:
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
    @with_tf_device_gpu
    def get_importance_vector(self, task_id, importance_criteria: str,
                              layer_separate=False, use_coreset=False) -> tuple or np.ndarray:
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
        self.excl_loss = None
        self.yhat = None
        self.build_model()

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        pass
