import tensorflow as tf

from DEN import DEN


class AFN(DEN):

    def __init__(self, den_config):
        super().__init__(den_config)

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
