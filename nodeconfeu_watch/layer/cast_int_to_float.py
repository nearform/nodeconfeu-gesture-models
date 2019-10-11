
import tensorflow as tf
import tensorflow.keras as keras

class CastIntToFloat(keras.layers.Layer):
    def __init__(self, normalize_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._normalize_factor = normalize_factor

    def call(self, inputs):
        return tf.cast(inputs, tf.dtypes.float32) / tf.constant(self._normalize_factor, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape
