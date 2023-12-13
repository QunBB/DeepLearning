/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <iostream>
#include <string>

#include "binary_code_hash.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T>
struct BinaryCodeHashFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out, int length, int t, bool succession) {
    // std::cout << "@@@@@@ Runnin CPU @@@@@@" << std::endl;
    // Compute Binary Code Hash.
    int block_num;
    int block_length;
    if (succession){
      block_num = (length - 1) / t + 1;
      block_length = t;
    } else {
      block_num = t + 1;
      block_length = (length - 1) / block_num + 1;
    }
    
    for (T i = 0; i < size; i++) {
      // Convert into binary
      int binary_code[length];
      T num = in[i];
      for(int k=0; k<length; k++){
          if (num > 0){
            binary_code[k] = num % 2;    
            num = num / 2; 
          } else {
            binary_code[k] = 0;
          }  
      }
      
      // Convert into 10base every block
      if (succession){
        for (int n = 0; n < block_num; n++){
          T num = 0;
          T start_index = n * (1 << block_length);
          for (int m = 0; m < t; m++){
            if (n*t+m>=length){
              break;
            }
            if (binary_code[n*t+m] == 1){
              num += 1 << m;
            }
          }
          out[i*block_num+n] = num + start_index;
        }
      }else { // skip
        for (int n = 0; n < block_num; n++){
          T num = 0;
          T start_index = n * (1 << block_length);
          for (int m = n; m < length; m+=t+1){
            if (binary_code[m] == 1){
              num += 1 << m;
            }
          }
          out[i*block_num+n] = num + start_index;
        }
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class BinaryCodeHashOp : public OpKernel {
 public:
  explicit BinaryCodeHashOp(OpKernelConstruction* context) : OpKernel(context) {
    // Check the inputs
    OP_REQUIRES_OK(context, context->GetAttr("length", &length_));
    OP_REQUIRES_OK(context, context->GetAttr("t", &t_));
    OP_REQUIRES_OK(context, context->GetAttr("strategy", &strategy_));

    OP_REQUIRES(context, length_ > 0,
                errors::InvalidArgument("Need length > 0, got ", length_));
    OP_REQUIRES(context, t_ > 0,
                errors::InvalidArgument("Need t > 0, got ", t_));
    OP_REQUIRES(context, length_ >= t_,
                errors::InvalidArgument("Need length >= t, got length: ", length_, " and t: ", t_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
    //             errors::InvalidArgument("BinaryCodeHash expects a 1-D vector."));

    // Create an output tensor
    int block_num;
    if (strategy_ == "succession"){
      block_num = (length_ - 1) / t_ + 1;
    } else {
      block_num = t_ + 1;
    }
    Tensor* output_tensor = NULL;
    tensorflow::TensorShape output_shape = input_tensor.shape();
    output_shape.AddDim(block_num);  // Add New dimension
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    BinaryCodeHashFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data(),
        length_, t_, strategy_ == "succession");
  }

  private:
    int length_;
    int t_;
    std::string strategy_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BinaryCodeHash").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BinaryCodeHashOp<CPUDevice, T>);
REGISTER_CPU(int64);
REGISTER_CPU(int32);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct BinaryCodeHashFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BinaryCodeHash").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BinaryCodeHashOp<GPUDevice, T>);
REGISTER_GPU(int32);
REGISTER_GPU(int64);

#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
