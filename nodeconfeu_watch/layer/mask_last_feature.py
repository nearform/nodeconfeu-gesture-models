
import tensorflow as tf
import tensorflow.keras as keras

class MaskLastFeature(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs[:, :, -1], 0)

    def call(self, inputs):
        return inputs[:, :, 0:-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] - 1)
