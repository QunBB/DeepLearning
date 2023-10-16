import tensorflow as tf
import numpy as np


d = np.arange(0,60).reshape([6, 10])

# 将array转化为tensor
data = tf.data.Dataset.from_tensor_slices(d)

# 从data数据集中按顺序抽取buffer_size个样本放在buffer中，然后打乱buffer中的样本
# buffer中样本个数不足buffer_size，继续从data数据集中安顺序填充至buffer_size，
# 此时会再次打乱
data = data.shuffle(buffer_size=3)

# 每次从buffer中抽取4个样本
data = data.batch(4)

# 将data数据集重复，其实就是2个epoch数据集
data = data.repeat(2)

# 构造获取数据的迭代器
iters = data.make_one_shot_iterator()

# 每次从迭代器中获取一批数据
batch = iters.get_next()

sess = tf.Session()

sess.run(batch)
# 数据集完成遍历完之后，继续抽取的话会报错：OutOfRangeError

"""
In [21]: d
Out[21]: 
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])
In [22]: sess.run(batch)
Out[22]: 
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

In [23]: sess.run(batch)
Out[23]: 
array([[40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])
"""

data = data.repeat(2)

data = data.shuffle(buffer_size=3)

data = data.batch(4)

"""
In [25]: sess.run(batch)
Out[25]: 
array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])

In [26]: sess.run(batch)
Out[26]: 
array([[50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])

In [27]: sess.run(batch)
Out[27]: 
array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])

"""