
import tensorflow as tf
import tensorflow.keras as keras

class MaskedConv(keras.layers.Layer):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.conv = keras.layers.Conv1D(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        masked_inputs = tf.where(tf.expand_dims(mask, -1), inputs, tf.constant(0, dtype=inputs.dtype))
        return self.conv(masked_inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
