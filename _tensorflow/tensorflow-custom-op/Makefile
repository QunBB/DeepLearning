CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

# 待编译的算子源码文件
BINARY_CODE_HASH_SRCS = tensorflow_binary_code_hash/cc/kernels/binary_code_hash_kernels.cc $(wildcard tensorflow_binary_code_hash/cc/kernels/*.h) $(wildcard tensorflow_binary_code_hash/cc/ops/*.cc)
BINARY_CODE_HASH_CPU_ONLY_SRCS = tensorflow_binary_code_hash/cc/kernels/binary_code_hash_only_cpu_kernels.cc $(wildcard tensorflow_binary_code_hash/cc/ops/*.cc)

# 获取tensorflow的c++源码位置
TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# 对于新版本的tensorflow, 需要使用新标准, 比如tensorflow2.10则需指定-std=c++17
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

# 编译目标so文件位置
BINARY_CODE_HASH_GPU_ONLY_TARGET_LIB = tensorflow_binary_code_hash/python/ops/_binary_code_hash_ops.cu.o
BINARY_CODE_HASH_TARGET_LIB = tensorflow_binary_code_hash/python/ops/_binary_code_hash_ops.so
BINARY_CODE_HASH_CPU_ONLY_TARGET_LIB = tensorflow_binary_code_hash/python/ops/_binary_code_hash_cpu_ops.so

# 编译命令: binary_code_hash op
binary_code_hash_gpu_only: $(BINARY_CODE_HASH_GPU_ONLY_TARGET_LIB)

$(BINARY_CODE_HASH_GPU_ONLY_TARGET_LIB): tensorflow_binary_code_hash/cc/kernels/binary_code_hash_kernels.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

binary_code_hash_op: $(BINARY_CODE_HASH_TARGET_LIB)
$(BINARY_CODE_HASH_TARGET_LIB): $(BINARY_CODE_HASH_SRCS) $(BINARY_CODE_HASH_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

binary_code_hash_cpu_only: $(BINARY_CODE_HASH_CPU_ONLY_TARGET_LIB)

$(BINARY_CODE_HASH_CPU_ONLY_TARGET_LIB): $(BINARY_CODE_HASH_CPU_ONLY_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

# Python调用测试
binary_code_hash_test: tensorflow_binary_code_hash/python/ops/binary_code_hash_ops_test.py tensorflow_binary_code_hash/python/ops/binary_code_hash_ops.py
	$(PYTHON_BIN_PATH) tensorflow_binary_code_hash/python/ops/binary_code_hash_ops_test.py

clean:
	rm -f $(BINARY_CODE_HASH_GPU_ONLY_TARGET_LIB) $(BINARY_CODE_HASH_TARGET_LIB) $(BINARY_CODE_HASH_CPU_ONLY_TARGET_LIB)
