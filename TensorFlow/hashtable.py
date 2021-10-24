import tensorflow as tf
import time
import numpy as np


def demo():
    """
    insert：插入键值对
    export：导出hashtable
    lookup：key查询
    remove：删除key
    size：hashtable的容量
    :return:
    """
    keys = tf.placeholder(dtype=tf.string, shape=[None])
    values = tf.placeholder(dtype=tf.int64, shape=[None])
    # 如果有多个表，则需要name命名，否则保存加载时，会因为都是默认命名而导致被覆盖
    table1 = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=-1,
                                                name="HashTable_1")
    table2 = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64, -1)
    insert_table1 = table1.insert(keys, values)
    insert_table2 = table2.insert(keys, values)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(insert_table1, feed_dict={keys: ["a"], values: [1]})
        sess.run(insert_table2, feed_dict={keys: ["b"], values: [2]})
        print("table1:", sess.run(table1.export()))
        print("table2:", sess.run(table2.export()))
        saver.saverve(sess, "checkpoint/test")


def run():
    """
    测试50W容量的hashtable，保存的大小和查询速度
    :return:
    """
    size = 500000
    keys = tf.placeholder(dtype=tf.string, shape=[None])
    values = tf.placeholder(dtype=tf.int64, shape=[None])
    # 如果有多个表，则需要name命名，否则保存加载时，会因为都是默认命名而导致被覆盖
    table1 = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=-1,
                                                name="tower/HashTable_1")
    insert_table1 = table1.insert(keys, values)
    lookup = table1.lookup(keys)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(insert_table1,
                 feed_dict={keys: ["id_" + str(i) for i in range(size)], values: list(range(size))})
        # print("table1:", sess.run(table1.export()))

        # 查询时间：0.007218122482299805
        # 模型大小：8.9M
        s1 = time.time()
        print(sess.run(lookup, feed_dict={keys: ["id_1", "id_100"]}))
        print(time.time() - s1)
        saver.save(sess, "checkpoint/test")


def test():
    """
    测试50W容量的hashtable，保存的大小和查询速度
    :return:
    """
    size = 500000
    keys = tf.placeholder(dtype=tf.string, shape=[None])
    values = tf.placeholder(dtype=tf.string, shape=[None])
    # 如果有多个表，则需要name命名，否则保存加载时，会因为都是默认命名而导致被覆盖
    table1 = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.string, default_value="",
                                                name="HashTable_1")
    insert_table1 = table1.insert(keys, values)
    lookup = table1.lookup(keys)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(insert_table1,
                 feed_dict={keys: ["id_" + str(i) for i in range(size)],
                            values: [np.array([i, i]).tostring() for i in range(size)]})
        # print("table1:", sess.run(table1.export()))

        # 查询时间：0.007218122482299805
        # 模型大小：8.9M
        s1 = time.time()
        print(sess.run(lookup, feed_dict={keys: ["id_1", "id_100"]}))
        print(time.time() - s1)
        saver.save(sess, "checkpoint/test")


def restore():
    # 如果有多个表，则需要name命名，否则保存加载时，会因为都是默认命名而导致被覆盖
    # 表名不受variable_scope的影响
    with tf.variable_scope("tower"):
        table1 = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=-1,
                                                    name="tower/HashTable_1")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "checkpoint/test")
        print(sess.run(table1.lookup(["id_1", "id_100"])))


if __name__ == '__main__':
    run()
    # test()
    restore()
