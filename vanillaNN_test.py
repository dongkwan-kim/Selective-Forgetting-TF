from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.misc

mnist = input_data.read_data_sets("DEN/MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
forget_task = 7
pruning = 3
init = tf.contrib.layers.xavier_initializer()

W1 = tf.get_variable("W1", shape=[784, 10], initializer=init)
W1_v = tf.Variable(W1)

y1 = tf.matmul(x, W1_v)

W2 = tf.get_variable("W2", shape=[10, 10], initializer=init)
W2_v = tf.Variable(W2)
y2 = tf.matmul(y1, W2_v)
y = tf.nn.softmax(y2)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
local_gradient = tf.gradients(cross_entropy, y1)
count = 0
label_name = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
acc_importance = np.zeros(10)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training section
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, loss, hidden, gradient = sess.run([train_step, cross_entropy, y1, local_gradient],
                                         feed_dict={x: batch_xs, y_: batch_ys})
    importance_vector = np.absolute(gradient * hidden)
    importance_vector = (importance_vector[0][0] * 255) / (max(importance_vector[0][0]) - min(importance_vector[0][0]))

# importance vector calculation
for i in range(1):
    for i in range(1000):
        count = count + 1
        batch_xs, batch_ys = mnist.train.next_batch(1)
        hidden, gradient = sess.run([y1, local_gradient], feed_dict={x: batch_xs, y_: batch_ys})
        importance_vector = np.absolute(gradient * hidden)  # importance vector

        if int(np.sum(label_name * batch_ys[0])) is not forget_task:
            acc_importance = importance_vector[0][0] + acc_importance
        # print(int(np.sum(label_name * batch_ys[0])))
    importance_vector = (importance_vector[0][0] * 255) / (max(importance_vector[0][0]) - min(importance_vector[0][0]))
    importance_vector2 = np.reshape(importance_vector, (2, 5))
    importance_vector2 = np.kron(importance_vector2, np.ones((10, 10)))
    scipy.misc.imsave('test2' + '_' + str(int(np.sum(label_name * batch_ys[0]))) + '_' + str(count) + '.jpg',
                      importance_vector2)
    sorted_index = np.argsort(acc_importance)

    print(acc_importance)
    sorted_index = sorted_index[0:pruning]
    print(sorted_index)
    temp_weight = W1_v.eval(session=sess)
    for i in sorted_index:
        for j in range(784):
            temp_weight[j][i] = 0
    W1_v.assign(temp_weight).eval(session=sess)
    importance_vector = np.reshape(acc_importance, (2, 5))

    importance_vector = np.kron(importance_vector, np.ones((10, 10)))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# accuracy check
print("total {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

print(f'forget_task: {forget_task}')
for j in range(10):
    acc_accuracy_result = 0
    count_v = 0
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(1)
        if int(np.sum(label_name * batch_ys[0])) == j:
            accuracy_result = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            acc_accuracy_result = accuracy_result + acc_accuracy_result
            count_v = count_v + 1

    print(j, ':', acc_accuracy_result / count_v)
