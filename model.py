
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras
from nodeconfeu_watch.reader import AccelerationDataset

tf.random.set_seed(0)

dataset = AccelerationDataset('./data/gestures-v1.csv')

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


inputs = keras.Input(shape=(None, 4), name='acceleration')
inputs_masked = MaskLastFeature()(inputs)
conv = keras.layers.Conv1D(10, 5, padding='causal')(inputs_masked)
normalized = keras.layers.LayerNormalization()(conv)
hidden = keras.layers.LSTM(10, return_sequences=True)(normalized)
outputs = keras.layers.LSTM(5)(hidden)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=dataset.train.x.shape[0],
                    epochs=87,
                    validation_data=(dataset.validation.x, dataset.validation.y))

print(
    sklearn.metrics.classification_report(
        dataset.test.y, tf.argmax(model.predict(dataset.test.x), -1).numpy(),
        target_names=dataset.classnames)
)
