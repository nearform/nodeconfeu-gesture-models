
import tensorflow as tf
import tensorflow.keras as keras

class DirectionFeatures(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        xy = tf.expand_dims(tf.sqrt(inputs[:, :, 0]**2 + inputs[:, :, 1]**2), -1)
        xz = tf.expand_dims(tf.sqrt(inputs[:, :, 0]**2 + inputs[:, :, 2]**2), -1)
        yz = tf.expand_dims(tf.sqrt(inputs[:, :, 1]**2 + inputs[:, :, 2]**2), -1)
        xyz = tf.norm(inputs, axis=-1, keepdims=1)

        return tf.concat(
            (inputs, xy, xz, yz, xyz),
            axis=-1
        )

    def compute_output_shape(self, input_shape):
        assert input_shape[2] == 3
        return (input_shape[0], input_shape[1], 7)
