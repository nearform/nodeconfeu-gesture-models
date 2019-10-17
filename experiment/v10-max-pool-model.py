
import numpy as np
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras

from nodeconfeu_watch.reader import AccelerationReader
from nodeconfeu_watch.layer import MaskLastFeature, GlobalMaxPooling, DirectionFeatures, MaskedConv
from nodeconfeu_watch.visual import plot_history

tf.random.set_seed(1)

dataset = AccelerationReader('./data/gestures-v2', test_ratio=0.2, validation_ratio=0.2,
                              max_sequence_length=50,
                              classnames=['swiperight', 'swipeleft', 'upup', 'waggle', 'clap2'],
                              input_dtype='float32',
                              input_shape='2d')

model = keras.Sequential()
model.add(keras.Input(shape=(50, 1, 4), name='acceleration', dtype=dataset.train.x.dtype))

model.add(keras.layers.Conv2D(14, (5, 1), padding='valid', activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(len(dataset.classnames), (5, 1), padding='same', dilation_rate=2, activation='relu'))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.MaxPool2D((46, 1)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(len(dataset.classnames), use_bias=False))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=64,
                    epochs=500,
                    validation_data=(dataset.validation.x, dataset.validation.y))

print(
    sklearn.metrics.classification_report(
        dataset.validation.y,
        tf.math.argmax(model.predict(dataset.validation.x), -1, output_type=tf.dtypes.int32).numpy(),
        target_names=dataset.classnames,
        digits=4)
)

print('making not quantized TFLite model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print(f'not quantized size: {len(tflite_model) / 1024}KB')
with open("exports/v10_not_quantized.tflite", "wb") as fp:
    fp.write(tflite_model)

print('making quantized TFLite model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_data_gen():
    indices = np.random.permutation(dataset.validation.x.shape[0])
    for input_value in dataset.validation.x[indices, ...]:
        yield [input_value[np.newaxis, :, :, :]]
converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()
print(f'quantized size: {len(tflite_model) / 1024}KB')
with open("exports/v10_quantized.tflite", "wb") as fp:
    fp.write(tflite_model)

plot_history(history)
