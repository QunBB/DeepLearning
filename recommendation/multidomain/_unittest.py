import unittest

import numpy as np
import os
import tensorflow as tf

from recommendation.utils.type_declaration import *
from recommendation.utils.interaction import Attention

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_multi_domain_inputs(num_domain):
    fields = [Field(name='goods_id', dim=64, vocabulary_size=10000),
              Field(name='shop_id', dim=64, vocabulary_size=100, l2_reg=0.00001),
              Field(name='user_id', dim=64, vocabulary_size=2000),
              Field(name='context', dim=64, vocabulary_size=200, l2_reg=0.00001),
              Field(name='domain_indicator', dim=64, vocabulary_size=num_domain, l2_reg=0.00001),
              ]
    inputs = dict(user_behaviors_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [32, 20])),
                                      'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [32, 20]))},
                  sequence_length=tf.convert_to_tensor(np.random.randint(1, 20, [32])),
                  target_ids={'goods_id': tf.convert_to_tensor(np.random.randint(0, 10000, [32])),
                              'shop_id': tf.convert_to_tensor(np.random.randint(0, 100, [32]))},
                  other_feature_ids={'user_id': tf.convert_to_tensor(np.random.randint(0, 2000, [32])),
                                     'context': tf.convert_to_tensor(np.random.randint(0, 200, [32]))},
                  domain_index=tf.convert_to_tensor(np.random.randint(0, num_domain))
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


if __name__ == '__main__':
    unittest.main()
