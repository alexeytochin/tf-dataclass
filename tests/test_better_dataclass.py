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
import unittest
from typing import Tuple

from tf_dataclass import dataclass
from tf_dataclass.get_type import get_output_type


@dataclass
class Base:
    x: int
    y: str


@dataclass
class Base2:
    z: int
    t: Base


class TestGetOutputType(unittest.TestCase):
    def test_tuple(self):
        def func() -> Tuple[int, str]:
            return (1, "abc")

        self.assertEqual(Tuple[int, str], get_output_type(func))


class TestFromTuple(unittest.TestCase):
    def test_tuple_simple(self):
        base = Base.from_tuple((1, 2))

        self.assertTrue(isinstance(base, Base))
        self.assertEqual(base.x, 1)
        self.assertEqual(base.y, 2)

    def test_tuple_nested(self):
        self.assertEqual(
            (1, (2, 'abc')),
            Base2(z=1, t=Base(x=2, y="abc")).as_tuple
        )