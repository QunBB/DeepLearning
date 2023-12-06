// kernel_example.h
#ifndef KERNEL_BINARY_CODE_HASH_H_
#define KERNEL_BINARY_CODE_HASH_H_

#include <string>

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct BinaryCodeHashFunctor {
  void operator()(const Device& d, int size, const T* in, T* out, int length, int t, bool succession);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_BINARY_CODE_HASH_H_
