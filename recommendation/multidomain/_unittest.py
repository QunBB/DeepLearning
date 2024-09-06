import unittest

import numpy as np
import os
import tensorflow as tf

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


def get_sar_net_inputs(num_scenario, batch_size=32, seq_len=20):
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

        fields, inputs = get_sar_net_inputs(num_scenario=self.num_domain)

        model = SARNet(fields,
                       num_scenario=self.num_domain,
                       num_scenario_experts=2,
                       num_shared_experts=6,
                       expert_inputs_dim=512,
                       fairness_coefficient_field_name='fairness_coefficient')

        output = model(**inputs)

        super().set_output(output)


if __name__ == '__main__':
    unittest.main()
