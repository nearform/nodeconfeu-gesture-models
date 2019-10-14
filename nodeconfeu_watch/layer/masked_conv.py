
import tensorflow as tf
import tensorflow.keras as keras

class MaskedConv(keras.layers.Layer):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.conv = keras.layers.Conv1D(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return self.conv(inputs)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)
