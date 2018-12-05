import tensorflow as tf
import numpy as np

from pprint import pprint

from DEN import DEN


class AFN(DEN):

    def __init__(self, den_config):
        super().__init__(den_config)
        self.afn_params = {}
        self.batch_idx = 0

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

    def clear(self):
        self.destroy_graph()
        self.sess.close()

    def train_den(self, flags, mnist, trainXs, valXs, testXs):
        params = dict()
        avg_perf = []

        for t in range(flags.n_tasks):
            data = (trainXs[t], mnist.train.labels,
                    valXs[t], mnist.validation.labels,
                    testXs[t], mnist.test.labels)

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
                temp_perf = self.predict_perform(j + 1, testXs[j], mnist.test.labels)
                temp_perfs.append(temp_perf)
            avg_perf.append(sum(temp_perfs) / float(t + 1))
            print("   [*] avg_perf: %.4f" % avg_perf[t])

            if t != flags.n_tasks - 1:
                self.clear()

    def predict_only_after_training(self, flags, mnist, testXs):
        print("\n PREDICT ONLY AFTER TRAINING")
        self.sess = tf.Session()
        temp_perfs = []
        for t in range(flags.n_tasks):
            temp_perf = self.predict_perform(t + 1, testXs[t], mnist.test.labels)
            temp_perfs.append(temp_perf)
        return temp_perfs

    def get_next_batch(self, x, y, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size
        next_idx = self.batch_idx + batch_size
        r = x[self.batch_idx:next_idx], y[self.batch_idx:next_idx]
        self.batch_idx = next_idx
        return r

    def remove_neurons(self, scope, indexes):
        w: tf.Variable = self.afn_get_variable(scope, "weight", True)
        b: tf.Variable = self.afn_get_variable(scope, "biases", True)

        val_w = w.eval(session=self.sess)
        val_b = b.eval(session=self.sess)

        for i in indexes:
            val_w[:, i] = 0
            val_b[i] = 0

        w = w.assign(val_w)
        b = b.assign(val_b)

        self.afn_params[w.name] = w
        self.afn_params[b.name] = b

        return w, b

    def get_importance_vector_with_training(self, task_id, mnist, trainXs, valXs, testXs):
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

        hidden_layer_list = []
        bottom = X
        stamp = self.time_stamp['task%d' % task_id]
        for i in range(1, self.n_layers):
            w = self.get_variable('layer%d' % i, 'weight', True)
            b = self.get_variable('layer%d' % i, 'biases', True)
            w = w[:stamp[i - 1], :stamp[i]]
            b = b[:stamp[i]]

            afn_w = self.afn_create_variable("afn_layer%d" % i, "weight", trainable=True, initializer=w)
            afn_b = self.afn_create_variable("afn_layer%d" % i, "biases", trainable=True, initializer=b)
            bottom = tf.nn.relu(tf.matmul(bottom, afn_w) + afn_b)
            hidden_layer_list.append(bottom)
            print(' [*] task %d, shape of layer %d : %s' % (task_id, i, afn_w.get_shape().as_list()))

        w = self.get_variable('layer%d' % self.n_layers, 'weight_%d' % task_id, True)
        b = self.get_variable('layer%d' % self.n_layers, 'biases_%d' % task_id, True)
        w = w[:stamp[self.n_layers - 1], :stamp[self.n_layers]]
        b = b[:stamp[self.n_layers]]
        afn_w = self.afn_create_variable("afn_layer%d" % self.n_layers, "weight_%d" % task_id,
                                         trainable=True, initializer=w)
        afn_b = self.afn_create_variable("afn_layer%d" % self.n_layers, "biases_%d" % task_id,
                                         trainable=True, initializer=b)

        y = tf.matmul(bottom, afn_w) + afn_b
        yhat = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y))

        train_step = tf.train.GradientDescentOptimizer(self.init_lr).minimize(loss)
        gradient_list = [tf.gradients(loss, h) for h in hidden_layer_list]

        self.sess.run(tf.global_variables_initializer())

        data = (trainXs[task_id], mnist.train.labels,
                valXs[task_id], mnist.validation.labels,
                testXs[task_id], mnist.test.labels)

        h_length = sum([h.get_shape().as_list()[-1] for h in hidden_layer_list])
        importance_vector = np.zeros(shape=(0, h_length))
        while True:
            batch_x, batch_y = self.get_next_batch(data[0], data[1])
            if len(batch_x) == 0:
                break

            hidden_1, hidden_2, gradient_1, gradient_2 = self.sess.run(
                hidden_layer_list + gradient_list,
                feed_dict={X: batch_x, Y: batch_y}
            )

            # Shape = Batch * |h|
            batch_importance_vector_1 = np.absolute(hidden_1 * gradient_1)[0]
            batch_importance_vector_2 = np.absolute(hidden_2 * gradient_2)[0]
            batch_importance_vector = np.concatenate((batch_importance_vector_1, batch_importance_vector_2), axis=1)
            importance_vector = np.vstack((importance_vector, batch_importance_vector))

        importance_vector = importance_vector.sum(axis=0)

        return importance_vector
