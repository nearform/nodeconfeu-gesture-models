
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras

from nodeconfeu_watch.reader import AccelerationDataset
from nodeconfeu_watch.layer import MaskLastFeature, DirectionFeatures, MaskedConv
from nodeconfeu_watch.visual import plot_history

tf.random.set_seed(1)

dataset = AccelerationDataset('./data/gestures-v1.csv', test_ratio=0, validation_ratio=0.25)

model = keras.Sequential()
model.add(keras.Input(shape=(None, 4), name='acceleration'))
model.add(MaskLastFeature())
model.add(DirectionFeatures())

model.add(MaskedConv(25, 1, padding='same'))

model.add(keras.layers.LayerNormalization(scale=False))
model.add(keras.layers.Activation('relu'))
model.add(MaskedConv(25, 3, padding='same'))

model.add(keras.layers.LayerNormalization(scale=False))
model.add(keras.layers.Activation('relu'))
model.add(MaskedConv(10, 3, padding='same', dilation_rate=3))

model.add(keras.layers.LayerNormalization(scale=False))
model.add(keras.layers.Activation('relu'))
model.add(MaskedConv(10, 3, padding='same', dilation_rate=9))

model.add(keras.layers.LSTM(10, use_bias=False))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(len(dataset.classnames), use_bias=False))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=dataset.train.x.shape[0],
                    epochs=100,
                    validation_data=(dataset.validation.x, dataset.validation.y))

print(
    sklearn.metrics.classification_report(
        dataset.validation.y,
        tf.argmax(model.predict(dataset.validation.x), -1).numpy(),
        target_names=dataset.classnames)
)

plot_history(history)
