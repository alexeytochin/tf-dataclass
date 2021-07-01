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
"""
tf-dataclass.

Support python dataclass containers as input and output in callable TensorFlow graph for tensorflow version >= 2.0.0.
"""
from tf_dataclass.modified_dataclass import dataclass, is_dataclass
from tf_dataclass.packing_tools import function


__version__ = "0.1.1"
__author__ = 'Alexey Tochin'
__all__ = ["dataclass", "is_dataclass", "function"]