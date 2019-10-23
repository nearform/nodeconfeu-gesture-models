
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras

from nodeconfeu_watch.reader import AccelerationReader
from nodeconfeu_watch.layer import MaskLastFeature, DirectionFeatures, MaskedConv
from nodeconfeu_watch.visual import plot_history, classification_report

tf.random.set_seed(0)

dataset = AccelerationReader({
        "gordon": './data/gordon-v1'
    },
    test_ratio=0, validation_ratio=0.25,
    classnames=['nothing', 'clap2', 'upup', 'swiperight', 'swipeleft'],
    max_sequence_length=None,
    input_shape='1d',
    mask_dimention=True)

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
model.add(keras.layers.LSTM(25, return_sequences=True))
model.add(keras.layers.LSTM(len(dataset.classnames)))
model.add(keras.layers.Dense(len(dataset.classnames), use_bias=False))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=dataset.train.x.shape[0],
                    epochs=200,
                    validation_data=(dataset.validation.x, dataset.validation.y))

print('')
print('Raw model performance on validation dataset')
print(classification_report(model, dataset, subset='validation'))

plot_history(history)
