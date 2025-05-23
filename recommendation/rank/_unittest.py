import unittest

import numpy as np
import os
from collections import OrderedDict
import tensorflow as tf

from recommendation.utils.type_declaration import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_tensor_inputs(sparse_list=['c1', 'c2'], dense_list=['d1']):
    batch_size = 32

    sparse_inputs_dict = OrderedDict()
    for name in sparse_list:
        sparse_inputs_dict[name] = tf.convert_to_tensor(np.random.randint(0, 99, size=[batch_size]))

    dense_inputs_dict = OrderedDict()
    for name in dense_list:
        dense_inputs_dict[name] = tf.convert_to_tensor(np.random.random(size=[batch_size]).astype(np.float32))
    return sparse_inputs_dict, dense_inputs_dict


def get_din_inputs(negative=False):
    fields = [DINField(name='goods_id', dim=64, vocabulary_size=10000,
                       mini_batch_regularization=True, ids_occurrence=np.random.randint(1, 100, [10000])),
              DINField(name='shop_id', dim=64, vocabulary_size=100, l2_reg=0.00001),
              DINField(name='user_id', dim=64, vocabulary_size=2000,
                       mini_batch_regularization=True, ids_occurrence=np.random.randint(1, 100, [2000])),
              DINField(name='context', dim=64, vocabulary_size=200, l2_reg=0.00001)
              ]
    inputs = dict(user_behaviors_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [32, 20])),
                                      'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [32, 20]))},
                  sequence_length=tf.convert_to_tensor(np.random.randint(1, 20, [32])),
                  target_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [32])),
                              'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [32]))},
                  other_feature_ids={'user_id': tf.convert_to_tensor(np.random.randint(0, 2000, [32])),
                                     'context': tf.convert_to_tensor(np.random.randint(0, 200, [32]))})
    if negative:
        inputs['negative_sequence_ids'] = {'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [32, 20])),
                                      'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [32, 20]))}

    return fields, inputs


def get_dsin_inputs():
    emb_dim = 64
    batch_size = 32
    max_seq_length = 20
    num_sessions = 10
    params = {'item_embedding_size': emb_dim * 2,
              'num_sessions': num_sessions,
              'seq_length_max': max_seq_length,
              }
    fields = [
        Field(name='goods_id', dim=emb_dim, vocabulary_size=10000),
        Field(name='shop_id', dim=emb_dim, vocabulary_size=100),
        Field(name='user_id', dim=emb_dim, vocabulary_size=2000),
        Field(name='age', dim=emb_dim, vocabulary_size=20)
    ]
    inputs = dict(
        user_session_behaviors_id=[
            {'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size, max_seq_length])),
             'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size, max_seq_length]))} for _ in
            range(num_sessions)
        ],
        seq_length_list=[
            tf.convert_to_tensor(np.random.randint(1, max_seq_length, [batch_size])) for _ in range(num_sessions)
        ],
        session_length=tf.convert_to_tensor(np.random.randint(1, num_sessions, [batch_size])),
        item_profile_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [32])),
                          'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [32]))},
        user_profile_ids={'user_id': tf.convert_to_tensor(np.random.randint(0, 2000, [32])),
                          'age': tf.convert_to_tensor(np.random.randint(0, 20, [32]))}
    )
    return fields, inputs, params


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
        from recommendation.rank import fms

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
        from recommendation.rank import fms
        from recommendation.rank import fnn

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
                        dnn_hidden_units=[256, 128],
                        fms_checkpoint='fms.ckpt'
                        )
        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestFiBiNet(BaseTestCase):

    def test(self):
        from recommendation.rank.fibinet import FiBiNet

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
        from recommendation.rank import pnn

        model = pnn.PNN(num_fields=20,
                        dim=64,
                        kernel_type=pnn.KernelType.Net,
                        micro_net_size=64,
                        add_inner_product=True,
                        add_outer_product=True,
                        product_layer_size=128,
                        dnn_hidden_units=[256, 128],
                        )

        output = model([tf.convert_to_tensor(np.random.random([32, 64]).astype(np.float32)) for _ in range(20)])

        super().set_output(output)


class TestDeepCrossing(BaseTestCase):

    def test(self):
        from recommendation.rank.deepcrossing import DeepCrossing

        model = DeepCrossing(residual_size=[256, 256, 256],
                             l2_reg=1e-5,
                             dropout=0.2)

        output = model([tf.convert_to_tensor(np.random.random([32, 64]).astype(np.float32)) for _ in range(20)])

        super().set_output(output)


class TestDCN(BaseTestCase):

    def test(self):
        from recommendation.rank.dcn import DCN

        model = DCN(input_dim=512,
                    cross_layer_num=3,
                    cross_network_type='matrix',
                    low_rank_dim=64,
                    dnn_hidden_units=[512, 128],
                    )

        output = model(tf.convert_to_tensor(np.random.random([32, 512]).astype(np.float32)))

        super().set_output(output)


class TestDeepFM(BaseTestCase):

    def test(self):
        from recommendation.rank.deepfm import DeepFM

        model = DeepFM([Field(name='c1', vocabulary_size=100),
                        Field(name='c2', vocabulary_size=100),
                        Field(name='d1')],
                       dnn_hidden_units=[256, 256],
                       linear_type=LinearTerms.LW,
                       model_type=FMType.FM)

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestNFM(BaseTestCase):

    def test(self):
        from recommendation.rank import nfm

        model = nfm.NFM([nfm.Field(name='c1', vocabulary_size=100, dim=10),
                         nfm.Field(name='c2', vocabulary_size=100, dim=10),
                         nfm.Field(name='d1', dim=10)],
                        linear_type=nfm.LinearTerms.FiLV,
                        dnn_hidden_units=[512, 128],
                        dnn_activation=tf.nn.relu)

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestxDeepFM(BaseTestCase):

    def test(self):
        from recommendation.rank.xdeepfm import xDeepFM

        model = xDeepFM([Field(name='c1', vocabulary_size=100, dim=10),
                         Field(name='c2', vocabulary_size=100, dim=10),
                         Field(name='d1', dim=10)],
                        cross_layer_sizes=[10, 20],
                        dnn_hidden_units=[128, 64],
                        dnn_activation=tf.nn.relu,
                        split_connect=True,
                        )

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestContextNet(BaseTestCase):

    def test(self):
        from recommendation.rank.contextnet import ContextNet

        model = ContextNet(num_block=3,
                           agg_dim=1024,
                           ffn_type='FFN')

        output = model(tf.convert_to_tensor(np.random.random([32, 10, 64]).astype(np.float32)))

        super().set_output(output)


class TestMaskNet(BaseTestCase):

    def test(self):
        from recommendation.rank.masknet import MaskNet

        model = MaskNet(agg_dim=1024,
                        num_mask_block=3,
                        mask_block_ffn_size=[256, 128, 64],
                        masknet_type='serial')

        # model = MaskNet(agg_dim=1024,
        #                 num_mask_block=3,
        #                 mask_block_ffn_size=256,
        #                 masknet_type='parallel',
        #                 hidden_layer_size=[256, 128])

        output = model(tf.convert_to_tensor(np.random.random([32, 10, 64]).astype(np.float32)))

        super().set_output(output)


class TestDIN(BaseTestCase):

    def test(self):
        from recommendation.rank.din import DIN

        fields, inputs = get_din_inputs()

        model = DIN(fields=fields,
                    mlp_hidden_units=[256, 128])

        output = model(**inputs)

        super().set_output(output)


class TestDIEN(BaseTestCase):

    def test(self):
        from recommendation.rank.dien import DIEN
        from recommendation.utils.interaction import AIGRU, AGRU, AUGRU

        fields, inputs = get_din_inputs(negative=True)

        model = DIEN(fields=fields,
                     mlp_hidden_units=[256, 128],
                     gru_hidden_size=256,
                     attention_gru=AIGRU)

        output = model(**inputs)

        super().set_output(output)


class TestDSIN(BaseTestCase):

    def test(self):
        from recommendation.rank.dsin import DSIN

        fields, inputs, params = get_dsin_inputs()

        model = DSIN(fields,
                     att_embedding_size=128,
                     lstm_hidden_size=128,
                     mlp_hidden_units=[256, 128],
                     num_attention_heads=4,
                     intermediate_size=256,
                     **params)

        output = model(**inputs)

        super().set_output(output)


class TestTIN(BaseTestCase):

    def test(self):
        from recommendation.rank.tin import TIN

        fields, inputs = get_din_inputs()

        model = TIN(fields=fields,
                     num_attention_heads=2,
                     attention_head_size=64,
                     hidden_units=[256, 64])

        output = model(**inputs)

        super().set_output(output)


class TestGwPFM(BaseTestCase):

    def test(self):
        from recommendation.rank.gwpfm import GwPFM

        model = GwPFM([Field(name='c1', vocabulary_size=100, dim=10, group='g1'),
                       Field(name='c2', vocabulary_size=100, dim=10, group='g1'),
                       Field(name='d1', dim=10, group='g2')],
                      dnn_hidden_units=[128, 64],
                      dnn_activation=tf.nn.relu,
                      )

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs()

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


class TestHMoE(BaseTestCase):

    def test(self):
        from recommendation.rank.hmoe import HMoE, Expert

        model = HMoE(expert_group={'group_1': {Expert.InnerProduct: {}, Expert.CrossNetwork: {'input_dim': 512}},
                                   'group_2': {Expert.SerialMaskNet: dict(agg_dim=1024, num_mask_block=3, mask_block_ffn_size=[256, 128, 64])}},
                      dnn_hidden_units=[128, 64],
                      dnn_activation=tf.nn.relu,
                      )

        output = model({'group_1': [tf.convert_to_tensor(np.random.random([32, 64]).astype(np.float32)) for _ in range(8)],
                        'group_2': [tf.convert_to_tensor(np.random.random([32, 64]).astype(np.float32)) for _ in range(8)]})

        super().set_output(output)


class TestAdaF2M2(BaseTestCase):

    def test(self):
        from recommendation.rank.adaf2m2 import AdaF2M2, Expert

        model = AdaF2M2(
            fields_list=[Field(name='user_id', vocabulary_size=100, dim=10),
                         Field(name='item_id', vocabulary_size=100, dim=10),
                         Field(name='context_id', vocabulary_size=100, dim=10)],
            state_id_fields=[Field(name='user_id', vocabulary_size=100, dim=10),
                             Field(name='item_id', vocabulary_size=100, dim=10)],
            state_non_id_fields=[Field(name='interaction_count')],
            num_sample=3,
            hidden_units=[128, 64],
            interaction=Expert.InnerProduct
        )

        sparse_inputs_dict, dense_inputs_dict = get_tensor_inputs(
            sparse_list=['user_id', 'item_id', 'context_id'],
            dense_list=['interaction_count']
        )

        output = model(sparse_inputs_dict, dense_inputs_dict)

        super().set_output(output)


if __name__ == '__main__':
    unittest.main()
