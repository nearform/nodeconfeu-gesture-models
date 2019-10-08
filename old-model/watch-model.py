
import sklearn.metrics
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import base64

tf.random.set_seed(0)

data = np.load('./data/gestures-v1.npz')
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']
classnames = list(data['classnames'])

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

history = model.fit(x_train, y_train,
                    batch_size=x_train.shape[0],
                    epochs=87,
                    validation_data=(x_test, y_test))

props = tf.nn.softmax(model.predict(x_test))

print(sklearn.metrics.classification_report(
  y_test, tf.argmax(props, -1).numpy(),
  target_names=classnames))

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
with open("export/model.tflite", "wb") as fp:
  fp.write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

# Save the model to disk
with open("export/model_quantized.tflite", "wb") as fp:
  fp.write(tflite_model)

print("var model=atob(\""+base64.b64encode(tflite_model)+"\");")
