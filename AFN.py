import tensorflow as tf
import numpy as np

from DEN import DEN


class AFN(DEN):

    def __init__(self, den_config):
        super().__init__(den_config)
        self.afn_params = {}

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

    def get_importance_vector_with_training(self, task_id, mnist, trainXs, valXs, testXs):
        print("\n GET IMPORTANCE VECTOR OF TASK %d" % task_id)

        X = tf.placeholder(tf.float32, [None, self.dims[0]])
        Y = tf.placeholder(tf.float32, [None, self.n_classes])

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
        local_gradient = tf.gradients(loss, bottom)

        self.sess.run(tf.global_variables_initializer())

        data = (trainXs[task_id], mnist.train.labels,
                valXs[task_id], mnist.validation.labels,
                testXs[task_id], mnist.test.labels)

        batch_x, batch_y = self.data_iteration(data[0], data[1], "Train")

        _, loss, hidden, gradient = self.sess.run([train_step, loss, bottom, local_gradient],
                                                  feed_dict={X: batch_x, Y: batch_y})

        importance_vector = np.absolute(gradient * hidden)
        
        return importance_vector
