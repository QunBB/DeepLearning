import unittest

import numpy as np
import os
from collections import OrderedDict
import tensorflow as tf

from Recommendation.Utils.type_declaration import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_tensor_inputs():
    batch_size = 32

    sparse_inputs_dict = OrderedDict(c1=tf.convert_to_tensor(np.random.randint(0, 99, size=[batch_size])),
                                     c2=tf.convert_to_tensor(np.random.randint(0, 99, size=[batch_size])))
    dense_inputs_dict = OrderedDict(d1=tf.convert_to_tensor(np.random.random(size=[batch_size]).astype(np.float32)))
    return sparse_inputs_dict, dense_inputs_dict


class BaseTestCase(unittest.TestCase):
    output = None

    @classmethod
    def setUpClass(cls) -> None:
        # 重置计算图
        tf.reset_default_graph()
        print('tensorflow: reset default graph')

    @classmethod
    def tearDownClass(cls) -> None:
        print('\ntensorflow result:')
        if cls.output is not None:
            print(cls.output)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            print(sess.run(cls.output))

    @classmethod
    def set_output(cls, output):
        cls.output = output


class TestFMs(BaseTestCase):

    def test(self):
        from Recommendation.RANK import fms

        model = fms.FMs([Field(name='c1', vocabulary_size=100),
                         Field(name='c2', vocabulary_size=100),
                         Field(name='d1')],
                        linear_type=fms.LinearTerms.FiLV,
                        model_type=fms.FMType.FEFM)

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestFNN(BaseTestCase):

    def test_fnn(self):
        from Recommendation.RANK import fms
        from Recommendation.RANK import fnn

        tf.logging.set_verbosity(tf.logging.INFO)

        def _fms_pretrain():
            sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

            model = fms.FMs([Field(name='c1', vocabulary_size=100),
                             Field(name='c2', vocabulary_size=100),
                             Field(name='d1')])
            output = model(sparse_inputs_dict, dense_inputs_dict)
            print(output)

            saver = tf.train.Saver()

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            print(sess.run(output))

            saver.save(sess, 'fms.ckpt')

        _fms_pretrain()

        # 重置计算图
        tf.reset_default_graph()

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        model = fnn.FNN([Field(name='c1', vocabulary_size=100),
                         Field(name='c2', vocabulary_size=100),
                         Field(name='d1')],
                        embedding_dim=4,
                        dnn_hidden_size=[256, 128],
                        fms_checkpoint='fms.ckpt'
                        )
        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestFiBiNet(BaseTestCase):

    def test(self):
        from Recommendation.RANK.fibinet import FiBiNet

        batch_size = 32

        model = FiBiNet(dnn_units=[512, 128],
                        dropout=0.2,
                        reduction_ratio=2,
                        num_groups=2,
                        bilinear_output_size=64,
                        bilinear_type='interaction',
                        bilinear_plus=False,
                        equal_dim=False, )
        output = model([tf.convert_to_tensor(np.random.random([batch_size, 128]).astype(np.float32)) for _ in range(20)],
                       [tf.convert_to_tensor(np.random.random([batch_size, 64]).astype(np.float32)) for _ in range(10)],
                       is_training=True)

        super().set_output(output)


class TestPNN(BaseTestCase):

    def test(self):
        from Recommendation.RANK import pnn

        model = pnn.PNN(num_fields=20,
                        dim=64,
                        kernel_type=pnn.KernelType.Net,
                        micro_net_size=64,
                        add_inner_product=True,
                        add_outer_product=True,
                        product_layer_size=128,
                        dnn_hidden_size=[256, 128],
                        )

        output = model([tf.convert_to_tensor(np.random.random([32, 64]).astype(np.float32)) for _ in range(20)])

        super().set_output(output)


class TestDeepCrossing(BaseTestCase):

    def test(self):
        from Recommendation.RANK.deepcrossing import DeepCrossing

        model = DeepCrossing(residual_size=[256, 256, 256],
                             l2_reg=1e-5,
                             dropout=0.2)

        output = model([tf.convert_to_tensor(np.random.random([32, 64]).astype(np.float32)) for _ in range(20)])

        super().set_output(output)


class TestDCN(BaseTestCase):

    def test(self):
        from Recommendation.RANK.dcn import DCN

        model = DCN(input_dim=512,
                    cross_layer_num=3,
                    cross_network_type='matrix',
                    low_rank_dim=64,
                    dnn_hidden_size=[512, 128],
                    )

        output = model(tf.convert_to_tensor(np.random.random([32, 512]).astype(np.float32)))

        super().set_output(output)


class TestDeepFM(BaseTestCase):

    def test(self):
        from Recommendation.RANK.deepfm import DeepFM

        model = DeepFM([Field(name='c1', vocabulary_size=100),
                        Field(name='c2', vocabulary_size=100),
                        Field(name='d1')],
                       dnn_hidden_size=[256, 256],
                       linear_type=LinearTerms.LW,
                       model_type=FMType.FM)

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestNFM(BaseTestCase):

    def test(self):
        from Recommendation.RANK import nfm

        model = nfm.NFM([nfm.Field(name='c1', vocabulary_size=100, dim=10),
                         nfm.Field(name='c2', vocabulary_size=100, dim=10),
                         nfm.Field(name='d1', dim=10)],
                        linear_type=nfm.LinearTerms.FiLV,
                        dnn_hidden_size=[512, 128],
                        dnn_activation=tf.nn.relu)

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestxDeepFM(BaseTestCase):

    def test(self):
        from Recommendation.RANK.xdeepfm import xDeepFM

        model = xDeepFM([Field(name='c1', vocabulary_size=100, dim=10),
                         Field(name='c2', vocabulary_size=100, dim=10),
                         Field(name='d1', dim=10)],
                        cross_layer_sizes=[10, 20],
                        dnn_hidden_size=[128, 64],
                        dnn_activation=tf.nn.relu,
                        split_connect=True,
                        )

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


if __name__ == '__main__':
    unittest.main()
