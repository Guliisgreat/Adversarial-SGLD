import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np


class Monitor(object):
    def __init__(self, name):
        super(Monitor, self).__init__()
        self.node_on_graph = tf.placeholder(tf.float32, name= name)
        self.variable_in_computation = tf.summary.scalar(name, self.node_on_graph)

        self.list_iteration_train = []
        self.list_iteration_point_estimate = []
        self.list_iteration_bayesian = []

        self.list_data_train = []
        self.list_data_point_estimate = []
        self.list_data_bayesian = []

        self.name = name



    def record_tensorboard(self, data, iteration, sess, object_writer):

        record = sess.run(self.variable_in_computation, feed_dict={self.node_on_graph: data})
        return object_writer.add_summary(record, iteration)


    def record_matplot(self, data, iteration, writer):

        if iteration >= 2000:
            if writer == 'train':
                self.list_data_train.append(data)
                self.list_iteration_train.append(iteration)
            if writer == 'point_estimation':
                self.list_data_point_estimate.append(data)
                self.list_iteration_point_estimate.append(iteration)
            if writer == 'bayesian':
                self.list_data_bayesian.append(data)
                self.list_iteration_bayesian.append(iteration)


    def save_plot_matplot(self, log_folder, iteration):
        plt.clf()
        plt.plot(self.list_iteration_train, self.list_data_train, 'b', label = 'training')
        plt.plot(self.list_iteration_point_estimate, self.list_data_point_estimate, 'g', label = 'point estimation')
        plt.plot(self.list_iteration_bayesian, self.list_data_bayesian, 'r', label = 'bayesian')

        plt.xlabel('iteration')
        plt.ylabel(self.name)
        # if self.name == 'cross_entropy':
        #     plt.legend(loc='upper right')
        # else:
        #     plt.legend(loc='lower right')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(log_folder, self.name + '_%g.png' % iteration))


    def save_result_numpy(self, log_folder):
        train_result = np.array([self.list_iteration_train, self.list_iteration_train])
        point_estimate_result = np.array([self.list_iteration_point_estimate, self.list_data_point_estimate])
        bayesian_result = np.array([self.list_iteration_bayesian, self.list_data_bayesian])

        outfile = os.path.join(log_folder, self.name)
        np.savez(outfile, train_result = train_result, point_estimate_result = point_estimate_result, bayesian_result = bayesian_result)



