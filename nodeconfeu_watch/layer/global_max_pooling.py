
import tensorflow as tf
import tensorflow.keras as keras

class GlobalMaxPooling(keras.layers.Layer):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)

    def call(self, inputs, mask=None):
        return tf.math.reduce_max(inputs, axis=tf.constant(1, shape=[1]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
