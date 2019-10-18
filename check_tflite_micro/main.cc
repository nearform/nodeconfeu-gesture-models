/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdio.h>

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


unsigned char* read_file(char* filepath) {
  // get file size
  FILE * file = fopen(filepath, "r+");
  if (file == nullptr) return nullptr;
  fseek(file, 0, SEEK_END);
  long int size = ftell(file);
  fclose(file);

  // Reading data to array of unsigned chars
  file = fopen(filepath, "r+");
  unsigned char* content = (unsigned char *) malloc(size);
  fread(content, sizeof(unsigned char), size, file);
  fclose(file);

  return content;
}

int main(int argc, char* argv[]) {
  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  if (argc != 2) {
    error_reporter->Report("./bad call. Usage: program file.tflite\n");
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  // const model_def = read_file()
  const unsigned char* model_data = read_file(argv[1]);
  const tflite::Model* model = ::tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::ops::micro::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // Finding the minimum value for your model may require some trial and error.
  const int tensor_arena_size = 60 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter.AllocateTensors();

  // Obtain pointers to the model's input and output tensors
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  // Fill data with zero
  size_t num_input_elements = input->bytes / sizeof(float);
  for (size_t i = 0; i < num_input_elements; i++) {
    input->data.f[i] = 0.0;
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
    exit(1);
  }

  // Read the predicted y value from the model's output tensor
  size_t num_output_elements = output->bytes / sizeof(float);
  for (size_t i = 0; i < num_output_elements; i++) {
    error_reporter->Report("output[%d] = %f\n", i, output->data.f[i]);
  }
}
