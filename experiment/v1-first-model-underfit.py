
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras

from nodeconfeu_watch.reader import AccelerationDataset
from nodeconfeu_watch.layer import MaskLastFeature

tf.random.set_seed(4)

dataset = AccelerationDataset('./data/gestures-v1.csv', test_ratio=0, validation_ratio=0.25)

inputs = keras.Input(shape=(None, 4), name='acceleration')
inputs_masked = MaskLastFeature()(inputs)
conv = keras.layers.Conv1D(10, 5, padding='causal')(inputs_masked)
normalized = keras.layers.LayerNormalization()(conv)
hidden = keras.layers.LSTM(10, return_sequences=True)(normalized)
outputs = keras.layers.LSTM(5)(hidden)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=dataset.train.x.shape[0],
                    epochs=97,
                    validation_data=(dataset.validation.x, dataset.validation.y))

print(
    sklearn.metrics.classification_report(
        dataset.validation.y,
        tf.argmax(model.predict(dataset.validation.x), -1).numpy(),
        target_names=dataset.classnames)
)
