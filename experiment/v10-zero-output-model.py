
import sklearn.metrics
import numpy as np
import tensorflow as tf


interpreter = tf.lite.Interpreter("exports/v10_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)
print(output_details)

interpreter.set_tensor(input_details["index"], np.zeros(input_details["shape"], dtype=input_details["dtype"]))
interpreter.invoke()
output = interpreter.get_tensor(output_details["index"])[0]
print(output)
