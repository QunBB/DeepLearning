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

#include <string>
#include <iostream>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

template <typename T>
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
    
    // std::cout << "length: " << length_ << ", t: " << t_ << ", strategy: " << strategy_ << std::endl;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
    //             errors::InvalidArgument("BinaryCodeHash expects a 1-D vector."));

    // Create an output tensor
    int block_num;
    int block_length;
    if (strategy_ == "succession"){
      block_num = (length_ - 1) / t_ + 1;
      block_length = t_;
    } else {
      block_num = t_ + 1;
      block_length = (length_ - 1) / block_num + 1;
    }
    Tensor* output_tensor = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
    //                                                  &output_tensor));
    // tensorflow::TensorShape output_shape({input_tensor.shape().dim_size(0), block_num});
    tensorflow::TensorShape output_shape = input_tensor.shape();
    output_shape.AddDim(block_num);  // Add New dimension
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    // Compute Binary Code Hash.
    const int N = input.size();
    // std::cout << "input size: " << N << ", output size: " << output_flat.size() << std::endl;
    for (int i = 0; i < N; i++) {
      // Convert into binary
      int binary_code[length_];
      T num = input(i);
      for(int k=0; k<length_; k++){
          if (num > 0){
            binary_code[k] = num % 2;    
            num = num / 2; 
          } else {
            binary_code[k] = 0;
          }  
      }
      
      // Convert into 10base every block
      if (strategy_ == "succession"){
        for (int n = 0; n < block_num; n++){
          T num = 0;
          T start_index = n * (1 << block_length);
          for (int m = 0; m < t_; m++){
            if (n*t_+m>=length_){
              break;
            }
            if (binary_code[n*t_+m] == 1){
              num += 1 << m;
            }
          }
          output_flat(i*block_num+n) = num + start_index;
        }
      }else { // skip
        for (int n = 0; n < block_num; n++){
          T num = 0;
          T start_index = n * (1 << block_length);
          for (int m = n; m < length_; m+=t_+1){
            if (binary_code[m] == 1){
              num += 1 << m;
            }
          }
          output_flat(i*block_num+n) = num + start_index;
        }
      }
    }
  }

  private:
    int length_;
    int t_;
    std::string strategy_;
};

// REGISTER_KERNEL_BUILDER(Name("BinaryCodeHash").Device(DEVICE_CPU).TypeConstraint<T>("T"), BinaryCodeHashOp<T>);
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BinaryCodeHash").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BinaryCodeHashOp<T>);
REGISTER_CPU(int64);
REGISTER_CPU(int32);
