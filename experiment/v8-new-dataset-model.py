
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras

from nodeconfeu_watch.reader import AccelerationReader
from nodeconfeu_watch.layer import MaskLastFeature, CastIntToFloat, DirectionFeatures, MaskedConv
from nodeconfeu_watch.visual import plot_history

tf.random.set_seed(1)

dataset = AccelerationReader('./data/gestures-v2', test_ratio=0.2, validation_ratio=0.2,
                              max_sequence_length=50,
                              classnames=['swiperight', 'swipeleft', 'upup', 'waggle', 'clap2'],
                              input_dtype='float32')

model = keras.Sequential()
model.add(keras.Input(shape=(50, 4), name='acceleration', dtype=dataset.train.x.dtype))
model.add(MaskLastFeature())
model.add(DirectionFeatures())

model.add(MaskedConv(14, 5, padding='same'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Activation('relu'))
model.add(MaskedConv(len(dataset.classnames), 3, padding='same', dilation_rate=2))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(len(dataset.classnames), use_bias=False))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
history = model.fit(dataset.train.x, dataset.train.y,
                    batch_size=64,
                    epochs=100,
                    validation_data=(dataset.validation.x, dataset.validation.y))

print(
    sklearn.metrics.classification_report(
        dataset.validation.y,
        tf.math.argmax(model.predict(dataset.validation.x), -1, output_type=tf.dtypes.int32).numpy(),
        target_names=dataset.classnames)
)

print('making not quantized TFLite model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print(f'not quantized size: {len(tflite_model) / 1024}KB')
with open("exports/v8_not_quantized.tflite", "wb") as fp:
    fp.write(tflite_model)

print('making quantized TFLite model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
print(f'quantized size: {len(tflite_model) / 1024}KB')
with open("exports/v8_quantized.tflite", "wb") as fp:
    fp.write(tflite_model)

plot_history(history)
