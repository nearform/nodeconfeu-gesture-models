
import sklearn.metrics
import numpy as np
import tensorflow as tf

from nodeconfeu_watch.reader import AccelerationReader

dataset = AccelerationReader('./data/gestures-v2', test_ratio=0.2, validation_ratio=0.2,
                              max_sequence_length=50,
                              classnames=['swiperight', 'swipeleft', 'upup', 'waggle', 'clap2'],
                              input_dtype='float32')

interpreter = tf.lite.Interpreter("exports/v7_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)
print(output_details)

predictions = []
for input_tensor in dataset.test.x:
    interpreter.set_tensor(input_details["index"], input_tensor[np.newaxis, ...])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predictions.append(np.argmax(output))

print(
    sklearn.metrics.classification_report(
        dataset.test.y,
        predictions,
        target_names=dataset.classnames)
)
