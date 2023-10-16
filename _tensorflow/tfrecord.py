import tensorflow as tf
import collections
import numpy as np

inputs_1 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
], dtype=np.int32)  # 这里的类型int32要与解析时使用的类型保持一致

inputs_2 = [
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6]
]
lables = [0, 1]


################################### 数据写入TFRecord ###################################

def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))  # 需要注意这里接受的格式是list，并且只能是一维的
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


def create_bytes_feature(values):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
    return f


writer = tf.python_io.TFRecordWriter('test.tfrecord')  # test.tfrecord是写入的文件路径

for i1, i2, l in zip(inputs_1, inputs_2, lables):
    features = collections.OrderedDict()  # 这里是新建一个有序字典
    # 对于多维数组，只能先将其转化为byte，才能传递给Feature
    features['inputs_1'] = create_bytes_feature(i1.tostring())
    features['inputs_2'] = create_float_feature(i2)
    features['labels'] = create_int_feature([l])
    features['test'] = create_bytes_feature("--test--".encode('utf-8'))

    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(example.SerializeToString())
writer.close()


################################### 解析TFrecord数据 ###################################

name_to_features = {
    "inputs_1": tf.FixedLenFeature([], tf.string),
    "inputs_2": tf.FixedLenFeature([3], tf.float32),  # 这里的格式需要与写入的保持一致，否则可能出现解析错误
    "labels": tf.FixedLenFeature([], tf.int64),
    "test": tf.FixedLenFeature([], tf.string)
}


# 这里还可以同时读取多个tfrecord文件
files = tf.gfile.Glob('*.tfrecord')
# [file_name_1, file_name_2, .....]
d = tf.data.TFRecordDataset(files)

# d = tf.data.TFRecordDataset('test.tfrecord')
d = d.repeat()  # 这里repeat不传递参数，则会无限重复
d = d.shuffle(buffer_size=2)
# map_and_batch其实就是map和batch结合在一起而已
d = d.apply(tf.contrib.data.map_and_batch(
    lambda record: tf.parse_single_example(record, name_to_features),
    batch_size=1))

iters = d.make_one_shot_iterator()
batch = iters.get_next()

# BytesList解析时会丢失shape信息，需要自己还原它的shape，所以一般也会将shape数据一同写入
# 这里需要将byte解析成原本的数据结构，这里的tf.int32需要与写入时的格式保持一致
inputs_1_batch = tf.decode_raw(batch['inputs_1'], tf.int32)  # tf.int32类型要与源数据的类型保持一致
inputs_1_batch = tf.reshape(inputs_1_batch, [-1, 2, 2])
# 因为每次是batch个inputs_1，所以shape是[-1, 2, 2]，原来的shape是[2, 2]
inputs_2_batch = batch['inputs_2']
labels_batch = batch['labels']

# 需decode('utf-8')
test_str = batch['test']

sess = tf.Session()
# 这样我们就可以每次获取一个batch的数据了
sess.run([inputs_1_batch, inputs_2_batch, labels_batch])
