import numpy as np
import tensorflow as tf

def lerp(a,b,w):
    return w*a + (1.-w)*b

class QNet(object):
    def __init__(self, in_dim):
        self.scope = tf.get_variable_scope().name
        shape = (None,) + in_dim
        self.inputs = tf.placeholder(shape=shape, dtype=tf.uint8)
        self.L = []

    def append(self, l):
        self.L.append(l)

    def setup(self):
        X = tf.contrib.layers.one_hot_encoding(self.inputs, 16) ## VERY SPECIFIC USE
        for l in self.L:
            X = l.apply(X)
        self._Q = X 

        self.predict = tf.argmax(self._Q, 1)
        self.actions = tf.placeholder(shape=[None,1], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, 4, dtype=tf.float32) #up-down-left-right

        self.Q = tf.reduce_sum(tf.multiply(self._Q, self.actions_one_hot), reduction_indices=1)
        self.Qn = tf.placeholder(shape=[None,1], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.Qn - self.Q))
        #trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

        self.update = trainer.minimize(loss)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def copyTo(self, net, tau):
        src = self.vars()
        dst = net.vars()
        return map(lambda (s,d): d.assign(lerp(s,d,tau)), zip(src,dst))
