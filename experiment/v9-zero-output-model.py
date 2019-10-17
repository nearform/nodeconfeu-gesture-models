
import sklearn.metrics
import numpy as np
import tensorflow as tf

from nodeconfeu_watch.reader import AccelerationReader

dataset = AccelerationReader('./data/gestures-v2', test_ratio=0.2, validation_ratio=0.2,
                              max_sequence_length=50,
                              classnames=['swiperight', 'swipeleft', 'upup', 'waggle', 'clap2'],
                              input_dtype='float32')

interpreter = tf.lite.Interpreter("exports/v9_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)
print(output_details)

interpreter.set_tensor(input_details["index"], np.zeros(input_details["shape"], dtype=input_details["dtype"]))
interpreter.invoke()
output = interpreter.get_tensor(output_details["index"])[0]
print(output)
