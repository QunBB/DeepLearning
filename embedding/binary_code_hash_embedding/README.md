# Binary Code Based Hash Embedding

Binary Code Based Hash Embedding原理详见: [专栏]()，其实现包括两部分：

1. 二进制码Hash的算子实现（基于tensorflow官方custom-op仓库制作: [git](https://github.com/tensorflow/custom-op)|[tutorial](https://www.tensorflow.org/guide/create_op)）

2. 基于二进制码的Hash Embedding的具体[Python实现](https://github.com/QunBB/DeepLearning/blob/main/embedding/binary_code_hash_embedding/binary_code_hash_embedding.py)

## 自定义算子

### tensorflow环境

**Ubuntu下，无需源码安装，pip安装的情况下已通过测试**

1. cuda与tensorflow之间版本已兼容，直接pip安装

2. cuda与tensorflow之间版本不兼容 

	a. 新建Python环境: 

	`conda create -n <your_env_name> python=<x.x.x> cudatoolkit=<x.x> cudnn -c conda-forge`

	b. 现有Python环境: 

	`conda install cudatoolkit=<x.x> cudnn -c conda-forge -n <your_env_name>`

	执行以上步骤后，再进行pip安装

3. 当然，你仍然可以选择源码编译安装: https://www.tensorflow.org/install/source

### 编译

```makefile
make clean
make binary_code_hash_op
```

执行完，会生成so文件: `tensorflow_binary_code_hash/python/ops/_binary_code_hash_ops.so`
`tensorflow_binary_code_hash/python/ops/_binary_code_hash_ops.cu.o`

如果你的机器没有GPU，则可以跳过CUDA编译，仅编译CPU版本

```makefile
make clean
make binary_code_hash_cpu_only
```

执行完，会生成so文件: `tensorflow_binary_code_hash/python/ops/_binary_code_hash_cpu_ops.so`

## Python实现

详见 [binary_code_hash_embedding.py](https://github.com/QunBB/DeepLearning/blob/main/embedding/binary_code_hash_embedding/binary_code_hash_embedding.py)

其中，算子引入方式如下：

```python
import tensorflow as tf

# for GPU and CPU
binary_code_hash_ops = tf.load_op_library('./tensorflow_binary_code_hash/python/ops/_binary_code_hash_ops.so')

# for only CPU
# binary_code_hash_ops = tf.load_op_library('./tensorflow_binary_code_hash/python/ops/_binary_code_hash_cpu_ops.so')

binary_code_hash = binary_code_hash_ops.binary_code_hash
```

## Issues

1. In file included from tensorflow_time_two/cc/kernels/time_two_kernels.cu.cc:21:0: /usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h:22:10: fatal error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory

如果是conda环境，tensorflow c++源码的头文件位置则在 `<your_anaconda_path>/envs/<your_env_name>/lib/pythonx.x/site-packages/tensorflow/include`
对于tensorflow 1.x，则不是存放在tensorflow，而是在tensorflow_core

解决方案一：

**Copy the CUDA header files to target directory. 拷贝CUDA头文件**

```shell
mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include && cp -r /usr/local/cuda/targets/x86_64-linux/include/* /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include
```

解决方案二：

**修改CUDA头文件.**

"tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h"

```c++
#include "third_party/gpus/cuda/include/cuda_fp16.h"
```

替换成

```c++
#include "cuda_fp16.h"
```

"tensorflow/include/tensorflow/core/util/gpu_device_functions.h"

```c++
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cuda.h"
```

替换成

```c++
#include "cuComplex.h"
#include "cuda.h"
```

2. tensorflow 2.x支持

对于新版本的tensorflow, **[Makefile](https://github.com/QunBB/DeepLearning/blob/main/embedding/binary_code_hash_embedding/Makefile#L14)中需要指定c++新标准**。 比如tensorflow2.10则需指定-std=c++17

3. tensorflow.python.framework.errors_impl.NotFoundError: dlopen(./tensorflow_binary_code_hash/python/ops/\_binary_code_hash_cpu_ops.so, 0x0006): Library not loaded: @rpath/libtensorflow_framework.2.dylib

运行Python脚本导入算子so文件时的错误。这种错误一般是**Python运行环境与编译时的Python环境不一致导致的**。