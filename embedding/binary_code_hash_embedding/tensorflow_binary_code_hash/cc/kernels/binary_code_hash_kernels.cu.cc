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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <array>
#include <stdio.h>
#include <iostream>
#include "binary_code_hash.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
// Cann't use c++ std.
template <typename T>
__global__ void BinaryCodeHashCudaKernel(const int size, const T* in, T* out, int length, int t, bool succession) {
  int block_num;
  int block_length;
  if (succession){
    block_num = (length - 1) / t + 1;
    block_length = t;
  } else {
    block_num = t + 1;
    block_length = (length - 1) / block_num + 1;
  }

  int* binary_code = new int[length];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    // out[i] = 2 * ldg(in + i);
    // Convert into binary
      T num = ldg(in + i);
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
  delete[] binary_code;
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct BinaryCodeHashFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out, int length, int t, bool succession) {
    // std::cout << "@@@@@@ Runnin CUDA @@@@@@" << std::endl;
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    BinaryCodeHashCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out, length, t, succession);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct BinaryCodeHashFunctor<GPUDevice, int32>;
template struct BinaryCodeHashFunctor<GPUDevice, int64>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
