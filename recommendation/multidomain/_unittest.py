import unittest

import numpy as np
import os
import tensorflow as tf
from numba.cpython.enumimpl import enum_eq

from recommendation.utils.type_declaration import *
from recommendation.utils.interaction import Attention

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_multi_domain_inputs(num_domain, batch_size=32, seq_len=20):
    fields = [Field(name='goods_id', dim=64, vocabulary_size=10000),
              Field(name='shop_id', dim=64, vocabulary_size=100, l2_reg=0.00001),
              Field(name='user_id', dim=64, vocabulary_size=2000),
              Field(name='context', dim=64, vocabulary_size=200, l2_reg=0.00001),
              Field(name='domain_indicator', dim=64, vocabulary_size=num_domain, l2_reg=0.00001),
              ]
    inputs = dict(user_behaviors_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size, seq_len])),
                                      'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size, seq_len]))},
                  sequence_length=tf.convert_to_tensor(np.random.randint(1, 20, [batch_size])),
                  target_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size])),
                              'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size]))},
                  other_feature_ids={'user_id': tf.convert_to_tensor(np.random.randint(0, 2000, [batch_size])),
                                     'context': tf.convert_to_tensor(np.random.randint(0, 200, [batch_size]))},
                  domain_index=tf.convert_to_tensor(np.random.randint(0, num_domain, [batch_size]))
                  )
    return fields, inputs


def get_star_net_inputs(num_scenario, batch_size=32, seq_len=20):
    fields = [Field(name='goods_id', dim=64, vocabulary_size=10000),
              Field(name='shop_id', dim=64, vocabulary_size=100, l2_reg=0.00001),
              Field(name='user_id', dim=64, vocabulary_size=2000),
              Field(name='scenario_id', dim=64, vocabulary_size=num_scenario),
              Field(name='scenario_type', dim=64, vocabulary_size=num_scenario // 2),
              Field(name='context', dim=64, vocabulary_size=200, l2_reg=0.00001),
              Field(name='fairness_coefficient', dim=64, vocabulary_size=num_scenario, l2_reg=0.00001),
              ]
    scenario_ids = tf.convert_to_tensor(np.random.randint(0, num_scenario, [batch_size]))
    inputs = dict(
        user_behaviors_items_sequence={
            'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size, seq_len])),
            'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size, seq_len]))},
        user_behaviors_scenario_context_sequence={
            'scenario_id': tf.convert_to_tensor(np.random.randint(0, num_scenario, [batch_size, seq_len])),
            'scenario_type': tf.convert_to_tensor(np.random.randint(0, num_scenario // 2, [batch_size, seq_len]))},
        sequence_length=tf.convert_to_tensor(np.random.randint(1, 20, [batch_size])),
        target_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size])),
                    'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size]))},
        scenario_context={
            'scenario_id': scenario_ids,
            'scenario_type': tf.convert_to_tensor(np.random.randint(0, num_scenario // 2, [batch_size]))},
        scenario_ids=scenario_ids,
        other_feature_ids={'user_id': tf.convert_to_tensor(np.random.randint(0, 2000, [batch_size])),
                           'context': tf.convert_to_tensor(np.random.randint(0, 200, [batch_size]))},
        fairness_coefficient=tf.convert_to_tensor(np.random.random([batch_size]).astype('float32'))
    )
    return fields, inputs


def get_multi_task_scenario_inputs(task_list, batch_size=32, num_sequence=2, seq_len=20, dim=16):
    inputs = dict(
        multi_history_embeddings_list=[tf.convert_to_tensor(np.random.random([batch_size, seq_len, dim]), tf.float32) for _ in range(num_sequence)],
        multi_history_len_list=[tf.convert_to_tensor(np.random.randint(1, seq_len, [batch_size]), tf.float32) for _ in range(num_sequence)],
        user_embeddings=tf.convert_to_tensor(np.random.random([batch_size, dim * 10]), tf.float32),
        scenario_embeddings=tf.convert_to_tensor(np.random.random([batch_size, dim * 2]), tf.float32),
        task_embeddings={task: tf.convert_to_tensor(np.random.random([batch_size, dim]), tf.float32) for task in task_list},
    )
    return inputs


def get_pepnet_inputs(domain_list, batch_size=32, seq_len=20):
    fields = [Field(name='goods_id', dim=64, vocabulary_size=10000),
              Field(name='shop_id', dim=64, vocabulary_size=100, l2_reg=0.00001),
              Field(name='user_id', dim=64, vocabulary_size=2000),
              Field(name='context', dim=64, vocabulary_size=200, l2_reg=0.00001),
              Field(name='domain_id', dim=64, vocabulary_size=len(domain_list), l2_reg=0.00001),
              ]
    inputs = dict(user_behaviors_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size, seq_len])),
                                      'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size, seq_len]))},
                  sequence_length=tf.convert_to_tensor(np.random.randint(1, 20, [batch_size])),
                  item_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [batch_size])),
                              'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [batch_size]))},
                  user_ids={'user_id': tf.convert_to_tensor(np.random.randint(0, 2000, [batch_size]))},
                  other_feature_ids={'context': tf.convert_to_tensor(np.random.randint(0, 200, [batch_size]))},
                  domain_ids={domain: {'domain_id': tf.convert_to_tensor(np.ones([batch_size], dtype="int32") * i)} for i, domain in enumerate(domain_list)},
                  )
    return fields, inputs


class BaseTestCase(unittest.TestCase):
    num_domain = 10
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


class TestSTAR(BaseTestCase):

    def test(self):
        from recommendation.multidomain.star import STAR

        fields, inputs = get_multi_domain_inputs(num_domain=self.num_domain)

        model = STAR(fields,
                     num_domain=self.num_domain,
                     attention_agg=Attention,
                     star_fcn_input_size=384,
                     star_fcn_units=[1024, 512, 256],
                     aux_net_hidden_units=[512, 128])

        output = model(**inputs)

        super().set_output(output)


class TestSARNet(BaseTestCase):

    def test(self):
        from recommendation.multidomain.sarnet import SARNet

        fields, inputs = get_star_net_inputs(num_scenario=self.num_domain)

        model = SARNet(fields,
                       num_scenario=self.num_domain,
                       num_scenario_experts=2,
                       num_shared_experts=6,
                       expert_inputs_dim=512,
                       fairness_coefficient_field_name='fairness_coefficient')

        output = model(**inputs)

        super().set_output(output)


class TestM2M(BaseTestCase):

    def test(self):
        from recommendation.multidomain.m2m import M2M

        num_sequence = 2
        max_len_sequence = 20
        dim = 16
        inputs = get_multi_task_scenario_inputs(['click', 'like'],
                                                num_sequence=num_sequence, seq_len=max_len_sequence, dim=dim)

        model = M2M(num_experts=6,
                    num_meta_unit_layer=2,
                    num_residual_layer=2,
                    num_attention_heads=2,
                    attention_head_size=256,
                    views_dim=256,
                    num_sequence=num_sequence,
                    max_len_sequence=max_len_sequence,
                    position_embedding_dim=dim
                    )

        inputs['compute_logit'] = True
        output = model(**inputs)

        super().set_output(output)


class TestPEPNet(BaseTestCase):

    def test(self):
        from recommendation.multidomain.pepnet import PEPNet

        fields, inputs = get_pepnet_inputs(['domain_1', 'domain_2'])

        model = PEPNet(fields=fields,
                       num_tasks=3)

        output = model(**inputs)

        super().set_output(output)


if __name__ == '__main__':
    unittest.main()
