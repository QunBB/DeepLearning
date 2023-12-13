# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for binary_code_hash ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import test
try:
  from tensorflow_binary_code_hash.python.ops.binary_code_hash_ops import binary_code_hash
except ImportError:
  from binary_code_hash_ops import binary_code_hash


class ZeroOutTest(test.TestCase):

  def testZeroOut(self):
    with self.test_session() as sess:

      if int(tf.__version__.split('.')[0]) == 1:  # tensorflow 1.x
        print(sess.run(binary_code_hash([9999, 16777216, 16777220, 16777300], length=24, t=7, strategy="succession")))
      else:  # tensorflow 2.x
        print(binary_code_hash([9999, 16777216, 16777220, 16777300], length=24, t=7, strategy="succession").numpy())


if __name__ == '__main__':
  test.main()
