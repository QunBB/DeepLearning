"""
Reference: https://github.com/weberrr/PE-LTR
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls
import time


seed = 3456
tf.set_random_seed(seed)
np.random.seed(seed)

batch_size = 2000
dim = 64

x = np.float32(np.random.rand(batch_size, dim))

# 回归
y = np.dot(x, np.random.rand(dim, 1)) + 0.3

# 二分类
y2 = np.random.randint(0, 2, [batch_size, 1])

# 不同task的loss的权重
weight_a = tf.placeholder(tf.float32, shape=[])
weight_b = tf.placeholder(tf.float32, shape=[])

# 共享的参数，可以借鉴GradNorm，仅使用最后一层共享网络的参数
with tf.variable_scope("shared_weight"):
    hidden = tf.layers.dense(x, dim // 2,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

with tf.variable_scope("task_a"):
    y_pre = tf.layers.dense(hidden, 1,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    y_pre = tf.squeeze(y_pre, axis=-1)
    loss_a = tf.reduce_mean(tf.square(y - y_pre))

with tf.variable_scope("task_b"):
    y2_pre = tf.layers.dense(hidden, 1,
                             activation=tf.nn.sigmoid,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    # y2_pre = tf.squeeze(y2_pre, axis=-1)
    loss_b = y2 * tf.log(y2_pre) + (1 - y2) * tf.log(y2_pre)
    loss_b = tf.negative(loss_b)
    loss_b = tf.reduce_mean(loss_b)

loss = weight_a * loss_a + weight_b * loss_b

optimizer = tf.train.GradientDescentOptimizer(0.1)

with tf.variable_scope("pareto"):
    a_gradients = []
    b_gradients = []
    for w in tf.trainable_variables(scope="shared_weight"):
        a_gradients.append(tf.reshape(tf.gradients(loss_a, w), [-1, 1]))
        b_gradients.append(tf.reshape(tf.gradients(loss_b, w), [-1, 1]))

    a_gradients = tf.concat(a_gradients, axis=0)
    b_gradients = tf.concat(b_gradients, axis=0)

train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def pareto_step(w, c, G):
    """
    ref:http://ofey.me/papers/Pareto.pdf
    K : the number of task
    M : the dim of NN's params
    :param W: # (K,1)
    :param C: # (K,1)
    :param G: # (K,M)
    :return:
    """
    GGT = np.matmul(G, np.transpose(G))  # (K, K)
    e = np.mat(np.ones(np.shape(w)))  # (K, 1)
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((np.transpose(e), np.mat(np.zeros((1, 1)))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    hat_w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), M), z)  # (K+1, 1)
    hat_w = hat_w[:-1]  # (K, 1)
    hat_w = np.reshape(np.array(hat_w), (hat_w.shape[0],))  # (K,)
    c = np.reshape(np.array(c), (c.shape[0],))  # (K,)
    new_w = ASM(hat_w, c)
    return new_w


def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0., None] for _ in range(len(hat_w))]
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    new_w = min_out.x + c
    return new_w


use_pareto = True
w_a, w_b = 0.5, 0.5
c_a, c_b = 0.4, 0.2
for step in range(0, 100):
    res = sess.run([a_gradients, b_gradients, train, loss, loss_a, loss_b],
                   feed_dict={weight_a: w_a, weight_b: w_b})

    if use_pareto:
        s = time.time()
        weights = np.mat([[w_a], [w_b]])
        paras = np.hstack((res[0], res[1]))
        paras = np.transpose(paras)
        w_a, w_b = pareto_step(weights, np.mat([[c_a], [c_b]]), paras)
        print("pareto cost: {}".format(time.time() - s))

    l, l_a, l_b = res[3:]
    print("step:{:0>2d} w_a:{:4f} w_b:{:4f} loss:{:4f} loss_a:{:4f} loss_b:{:4f} r:{:4f}".format(
        step, w_a, w_b, l, l_a, l_b, l_a / l_b))
