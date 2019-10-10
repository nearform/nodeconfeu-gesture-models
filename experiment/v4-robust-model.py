
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
model.add(MaskedConv(50, 5, padding='causal'))
model.add(keras.layers.LayerNormalization())
# Adding dropout here to solve a local minimal problem. This problem exists
# because the dataset is so small, that mini-batches can't be used. Thus the
# loss-curvature is non-stocastic.
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(25, return_sequences=True, activation='relu')))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.SimpleRNN(25, use_bias=False))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(len(dataset.classnames), use_bias=False))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=dataset.train.x.shape[0],
                    epochs=200,
                    validation_data=(dataset.validation.x, dataset.validation.y),
                    callbacks=[
                        keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 if epoch > 470 else 1e-3)
                    ])

print(
    sklearn.metrics.classification_report(
        dataset.validation.y,
        tf.argmax(model.predict(dataset.validation.x), -1).numpy(),
        target_names=dataset.classnames)
)

plot_history(history)
