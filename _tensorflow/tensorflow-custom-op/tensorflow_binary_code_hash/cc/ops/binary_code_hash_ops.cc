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
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BinaryCodeHash")
    .Attr("T: {int64, int32}")
    .Input("hash_id: T")
    .Attr("length: int")
    .Attr("t: int")
    .Attr("strategy: {'succession', 'skip'}")
    .Output("bh_id: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int length;
      int t;
      std::string strategy;
      c->GetAttr("length", &length);
      c->GetAttr("t", &t);
      c->GetAttr("strategy", &strategy);
      int block_num;
      if (strategy == "succession"){
        block_num = (length - 1) / t + 1;
      } else {
        block_num = t + 1;
      }

      // 获取输入张量的形状
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input_shape));

      // 获取输入张量的维度数
      int input_rank = c->Rank(input_shape);

      // 创建新的形状列表
      std::vector<shape_inference::DimensionHandle> output_shape;
      for (int i = 0; i < input_rank; ++i) {
          output_shape.push_back(c->Dim(input_shape, i));
      }

      // 添加一个额外的维度
      output_shape.push_back(c->MakeDim(block_num));

      // 将output_shape转换为输出张量的形状
      c->set_output(0, c->MakeShape(output_shape));

      return Status::OK();
    });
