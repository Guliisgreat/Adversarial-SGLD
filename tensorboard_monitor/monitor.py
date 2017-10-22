import tensorflow as tf

class Monitor(object):
    def __init__(self, name):
        super(Monitor, self).__init__()
        self.node_on_graph = tf.placeholder(tf.float32, name= name)
        self.variable_in_computation = tf.summary.scalar(name, self.node_on_graph)


    def record_tensorboard(self, data, iteration, sess, object_writer):

        record = sess.run(self.variable_in_computation, feed_dict={self.node_on_graph: data})
        return object_writer.add_summary(record, iteration)
