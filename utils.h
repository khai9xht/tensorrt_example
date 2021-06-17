#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<NvInfer.h>
#include<NvOnnxParser.h>
#include<vector>
#include<cuda_runtime_api.h>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/core/cuda/warp.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/core.hpp>
#include<algorithm>
#include<numeric>


struct TRTDestroy{
  template <class T>
  void operator()(T* obj) const{
    if (obj){
      obj->destroy();
    }
  }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

size_t getSizeByDim(const nvinfer1::Dims& dims){

}

std::vector<std::string> getClassNames(const std::string& imagenet_classes){

}

void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims){

}

void postprocessResults(float* gpu_output, const nvinfer1::Dims &dims, int batch_size){

}

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context){

}

#endif
