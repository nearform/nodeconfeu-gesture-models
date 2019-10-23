
import math
import numpy as np
import tensorflow as tf

from .tflite_schema import Model as tflite_schema_model
from .tflite_schema import BuiltinOperator as tflite_schema_builtin_operator
from .tflite_schema import TensorType as tflite_schema_tensor_type

builtin_operator_code_lookup = {
    code: name
    for name, code
    in vars(tflite_schema_builtin_operator.BuiltinOperator).items()
    if not name.startswith('_')
}

tensor_type_code_lookup = {
    code: name
    for name, code
    in vars(tflite_schema_tensor_type.TensorType).items()
    if not name.startswith('_')
}

tensor_type_bits = {
  "FLOAT32": 32,
  "FLOAT16": 16,
  "INT32": 32,
  "UINT8": 8,
  "INT64": 64,
  "STRING": np.nan,
  "BOOL": 1,
  "INT16": 16,
  "COMPLEX64": 64,
  "INT8": 8
}

builtin_operator_version_support = {
  "DEPTHWISE_CONV_2D": [1],
  "FULLY_CONNECTED": [1,2,3,4],
  "MAX_POOL_2D": [1, 2],
  "SOFTMAX": [1],
  "LOGISTIC": [1],
  "SVDF": [1],
  "CONV_2D": [1, 2, 3],
  "AVERAGE_POOL_2D": [1],
  "ABS": [1],
  "SIN": [1],
  "COS": [1],
  "LOG": [1],
  "SQRT": [1],
  "RSQRT": [1],
  "SQUARE": [1],
  "PRELU": [1],
  "FLOOR": [1],
  "MAXIMUM": [1],
  "MINIMUM": [1],
  "ARG_MAX": [1],
  "ARG_MIN": [1],
  "LOGICAL_OR": [1],
  "LOGICAL_AND": [1],
  "LOGICAL_NOT": [1],
  "RESHAPE": [1],
  "EQUAL": [1],
  "NOT_EQUAL": [1],
  "GREATER": [1],
  "GREATER_EQUAL": [1],
  "LESS": [1],
  "LESS_EQUAL": [1],
  "CEIL": [1],
  "ROUND": [1],
  "STRIDED_SLICE": [1],
  "PACK": [1],
  "SPLIT": [1,2,3],
  "UNPACK": [1],
  "NEG": [1],
  "ADD": [1],
  "QUANTIZE": [1,2,3,4],
  "DEQUANTIZE": [1,2,3,4]
}


def _export_model(model, dataset, quantize):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: (
            [obs] for obs in dataset.validation.x[
                np.random.permutation(dataset.validation.x.shape[0]),
                np.newaxis,
                ...]
        )
    return converter.convert()


def _validate_flatbuffer_for_tflite_micro(model_bytes):
    buffer = bytearray(model_bytes)
    model = tflite_schema_model.Model.GetRootAsModel(buffer, 0)

    if model.Version() < 3:
        raise ValueError('only version 3 of the TFLite format is supported, use TensorFlow 2')

    for operator_code_index in range(model.OperatorCodesLength()):
        operator_code = model.OperatorCodes(operator_code_index)

        if operator_code.CustomCode() is not None:
            raise ValueError('Custom operators are not supported')

        operator_name = builtin_operator_code_lookup[operator_code.BuiltinCode()]
        if operator_name not in builtin_operator_version_support:
            raise ValueError(f'the operator {operator_name} is not supported by TFLite Micro')

        operator_version = operator_code.Version()
        if operator_version not in builtin_operator_version_support[operator_name]:
            raise ValueError(f'the operator {operator_name} does not support version {operator_version}')

    if model.SubgraphsLength() != 1:
        raise ValueError('expected only 1 subgraph')
    graph = model.Subgraphs(0)

    for tensor_index in range(graph.TensorsLength()):
        tensor = graph.Tensors(tensor_index)
        tensor_shape = tensor.ShapeAsNumpy()
        tensor_name = tensor.Name().decode()

        if tensor_shape.size == 0:
            raise ValueError(f'shape not specified for tensor {tensor_index} ({tensor_name})')


class ExportModel:
    def __init__(self, model, dataset, quantize=True, assert_export=True):
        self._quantize = quantize
        self._model_bytes = _export_model(model, dataset, quantize)
        if assert_export:
            self.evaluate_zeros_input()
            _validate_flatbuffer_for_tflite_micro(self._model_bytes)

    def modelsize(self):
        return len(self._model_bytes)

    def areasize(self):
        total_areasize = 0
        model = tflite_schema_model.Model.GetRootAsModel(bytearray(self._model_bytes), 0)

        graph = model.Subgraphs(0)
        for tensor_index in range(graph.TensorsLength()):
            tensor = graph.Tensors(tensor_index)
            tensor_shape = tensor.ShapeAsNumpy()
            tensor_type = tensor_type_code_lookup[tensor.Type()]

            tensor_size_bits = np.prod(tensor_shape) * tensor_type_bits[tensor_type]
            tensor_size_bytes = math.ceil(tensor_size_bits / 32) * 4 # round up to nearst 32bit-word
            total_areasize += tensor_size_bytes

        return total_areasize

    def size_report(self):
        modelsize = self.modelsize()
        areasize = self.areasize()
        total = modelsize + areasize

        return (
            f'{"Quantized" if self._quantize else "Not-quantized"} model\n'
            f'  modelsize {modelsize / 1024}KB\n'
            f'  areasize: ~{areasize / 1024}KB\n'
            f'  total: ~{total / 1024}KB'
        )

    def save(self, filepath):
        with open(filepath, "wb") as fp:
            fp.write(self._model_bytes)

    def evaluate_zeros_input(self):
        interpreter = tf.lite.Interpreter(model_content=self._model_bytes)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        interpreter.set_tensor(
            input_details["index"],
            np.zeros(input_details["shape"], dtype=input_details["dtype"]))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        return output

    def predict(self, input_tensor):
        interpreter = tf.lite.Interpreter(model_content=self._model_bytes)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        predictions = []
        for observation in input_tensor:
            interpreter.set_tensor(input_details["index"], observation[np.newaxis, ...])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            predictions.append(output)

        return np.stack(predictions)