# Copyright 2021 Alexey Tochin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import unittest
from typing import Tuple
import tensorflow as tf

from tf_dataclass import dataclass, function


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class Base:
    x: tf.Tensor
    y: tf.Tensor


@dataclass
class Base2:
    z: tf.Tensor
    t: Base


class TestTfDataclass(unittest.TestCase):
    def test_single_dataclass_output(self):
        @function
        def func(a) -> Base:
            return Base(x=a + 1, y=tf.constant(2))

        func(a=2)

    def test_tuple_output(self):
        @function
        def func(a) -> Tuple[tf.Tensor, tf.Tensor]:
            return a + 1, a + 2

        func(a=2)

    def test_tuple_output_of_tensor_and_dataobject(self):
        @function
        def func(a) -> Tuple[tf.Tensor, Base]:
            return a + 1, Base(x=a, y=1)

        func(a=2)

    def test_hierarchic_dataobject(self):
        @function
        def func(a) -> Base2:
            return Base2(z=a + 1, t=Base(x=a, y=1))

        func(a=2)

    def test_single_dataclass_input(self):
        @function
        def func(base: Base) -> tf.Tensor:
            return base.x

        self.assertTrue(isinstance(func(base=Base(x=1, y=2)), tf.Tensor))

    def test_dataclass_input_and_output(self):
        @function
        def func(base2: Base2) -> Base:
            return base2.t

        output = func(base2=Base2(z=1, t=Base(x=2, y=3)))

        self.assertTrue(isinstance(output, Base))


    def test_dataclass_input_and_output_no_arg_type_hint(self):
        @function
        def func(base2) -> Base:
            return base2.t

        output = func(base2=Base2(z=1, t=Base(x=2, y=3)))

        self.assertTrue(isinstance(output, Base))

    def test_sequential_feature(self):
        @dataclass
        class Sequential:
            feature: tf.Tensor  # shape = [batch, length, channels],    dtype = tf.float32
            length: tf.Tensor   # shape = [batch],                      dtype = tf.int32

        @function
        def convolution(input: Sequential, filters: tf.Tensor, stride: int) -> Sequential:
            return Sequential(
                feature = tf.nn.conv1d(input.feature, filters, stride, padding="SAME"),
                length = tf.math.floordiv(input.length, stride),
            )

        output = convolution(
            input = Sequential(
                feature = tf.random.normal(shape=[2, 6, 3]),
                length = tf.constant([6, 4], dtype=tf.int32),
            ),
            filters = tf.random.normal(shape=[1, 3, 7]),
            stride = 2,
        )

        self.assertTrue(isinstance(output, Sequential))
        self.assertEqual(list(output.feature.shape), [2, 3, 7])
        self.assertEqual(output.length.numpy().tolist(), [3, 2])
