
import tensorflow as tf
import tensorflow.keras as keras

class DirectionFeatures(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        xy = tf.norm(inputs[:, :, 0:2], axis=-1, keepdims=1)
        xz = tf.norm(tf.concat((inputs[:, :, 0:1], inputs[:, :, 2:3]), axis=-1), axis=-1, keepdims=1)
        yz = tf.norm(inputs[:, :, 1:3], axis=-1, keepdims=1)
        xyz = tf.norm(inputs, axis=-1, keepdims=1)

        return tf.concat(
            (inputs, xy, xz, yz, xyz),
            axis=-1
        )

    def compute_output_shape(self, input_shape):
        assert input_shape[2] == 3
        return (input_shape[0], input_shape[1], 7)
