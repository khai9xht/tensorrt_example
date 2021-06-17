#include <iostream>
#include "utils.h"

int main(int argc, char* argv[]){
  if (argc < 3){
    std::cerr << "usage: " << argv[0] << "model.onnx image.jpg\n";
    return -1;
  }
  std::string model_path(argv[1]);
  std::string image_path(argv[2]);
  int batch_size = 1;

  // initailize TensorRT engine and parse ONNX model
  TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
  TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
  parseOnnxModel(model_path, engine, context);
  return 0;
}
