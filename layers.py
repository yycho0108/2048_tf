import tensorflow as tf

class Layer(object):
    cnt = 0
    def __init__(self):
        pass
    def apply(self,x):
        pass
    @staticmethod
    def name(prefix):
        Layer.cnt += 1
        return str(prefix) + '_' + str(Layer.cnt)

class DenseLayer(Layer):
    def __init__(self, shape):
        super(DenseLayer,self).__init__()
        self.W = tf.get_variable(Layer.name('W'), shape = shape, initializer = tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable(Layer.name('b'), initializer = tf.zeros((shape[-1],)))

    def apply(self, x):
        x = tf.contrib.layers.flatten(x)
        return tf.matmul(x, self.W) + self.b


class ActivationLayer(Layer):
    def __init__(self,t):
        super(ActivationLayer,self).__init__()
        self.type = t

    def apply(self,x):
        if self.type == 'relu':
            return tf.nn.relu(x)
        elif self.type == 'softmax':
            return tf.nn.softmax(x)
        elif self.type == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif self.type == 'tanh':
            return tf.nn.tanh(x)
        else:
            #elif self.type == 'linear':
            return x

class ConvolutionLayer(Layer):
    def __init__(self, shape):
        #(?,4,4,16) --> ... (?,4,4,OUTPUT_CHANNELS)
        # shape = (patch_i, patch_j, input_chan, output_chan)
        self.W = tf.get_variable(Layer.name('W'), shape = shape, initializer = tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable(Layer.name('b'), initializer = tf.zeros((shape[-1],)))
    def apply(self, x):
        return tf.nn.conv2d(x, self.W, strides=[1,1,1,1], padding='SAME') + self.b
