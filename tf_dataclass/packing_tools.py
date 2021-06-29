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
from collections import Callable
from typing import Any, Dict, Optional
import tensorflow as tf

from tf_dataclass.get_type import get_output_type, get_input_type_dict
from tf_dataclass.modified_dataclass import is_dataclass


def unpack(value: Any, temple: Optional[type] = None) -> Any:
    if temple is None:
        temple = type(value)

    if is_dataclass(temple):
        return value.as_tuple
    elif temple == tuple or (hasattr(temple, "__origin__") and temple.__origin__ == tuple):
        return tuple(map(lambda sub_value, sub_temple: unpack(sub_value, sub_temple), value, temple.__args__))
    else:
        return value


def pack(unpacked_value: Any, temple: type) -> Any:
    if is_dataclass(temple):
        return temple.from_tuple(data_tuple=unpacked_value)
    else:
        return unpacked_value


def pack_function(func: Callable, input_type_dict: Dict[str, type], output_type: type):
    """
    Returns a version of @param func where its input and outputs are replaced by their unpacked versions.
        func -> pack . func . unpack
    @param func: input1, input2, ... -> output
    @return: input1_tuple, input2_tuple, ... -> output_tuple
    """
    def dictorized_func(**kwargs):
        assert kwargs.keys() == input_type_dict.keys()
        packed_arg_dict = {
            arg_name: pack(unpacked_value=kwargs[arg_name], temple=type_val)
            for arg_name, type_val in input_type_dict.items()
        }
        output = func(**packed_arg_dict)
        unpacked_output = unpack(value=output, temple=output_type)
        return unpacked_output

    return dictorized_func


def unpack_function(packed_func: Callable, input_type_dict: Dict[str, type], output_type: type):
    """
    Returns a version of @param func where its input and outputs are replaced by their unpacked versions. 
        func -> unpack . func . pack    
    @param packed_func: input1_tuple, input2_tuple, ... -> output_tuple
    @return: input1, input2, ... -> output
    """
    def undictorized_func(*args, **kwargs):
        if args:
            raise ValueError("Only keyword arguments are currently supported.")
        assert kwargs.keys() == input_type_dict.keys()
        input_kwargs = {}
        for arg_name, arg_value in kwargs.items():
            unpacked_arg = unpack(value=arg_value, temple=input_type_dict[arg_name])
            input_kwargs[arg_name] = unpacked_arg
        output_dict = packed_func(**input_kwargs)
        output = pack(unpacked_value=output_dict, temple=output_type)
        return output
    return undictorized_func


def function(func: Callable, **kwargs) -> Callable:
    """
    Modification of tensorflow.function for dataclass input/output support.
    1. dataclass decorator must be imported form tf_dataclass module
    2. Type hint for @parm func return type is mandatory
    3. Only keword arguments for the returned function are currently supported.
    4. Other arguments are the same as for tensorlfow.function
    See https://github.com/alexeytochin/tf-dataclass/blob/main/README.md for further details.

    @param func: the same as for tensorflow.function but requires typehints for the return type.
    @param kwargs: this argumets are pathed to tensorflow.function
    @return: callable object that accepts dataclass objects as input and/or output.
        Only keyword arguments for the decorated function are currently supported

    Example 1:
    >>> from tf_dataclass import dataclass, function
    >>> @dataclass
    >>> class Sequential:
    >>>     feature: tf.Tensor  # shape = [batch, length, channels],    dtype = tf.float32
    >>>     length: tf.Tensor   # shape = [batch],                      dtype = tf.int32
    >>> input = Sequential(
    >>>     feature = tf.random.normal(shape=[2, 6, 3]),
    >>>     length = tf.constant([6, 4], dtype=tf.int32),
    >>> )
    >>> @function
    >>> def convolution(input: Sequential, filters: tf.Tensor, stride: int) -> Sequential:
    >>>     return Sequential(
    >>>         feature = tf.nn.conv1d(input.feature, filters, stride),
    >>>         length = tf.math.floordiv(input.length, stride),
    >>>     )
    >>> output = convolution(
    >>>     input = input,
    >>>     filters = tf.random.normal(shape=[1, 3, 7]),
    >>>     stride = 2,
    >>> )
    >>> assert isinstance(output, Sequential)
    >>> print(output.length) # -> tf.Tensor([3 2], shape=(2,), dtype=int32)

    Example 2:
    >>> from typing import Tuple
    >>> from tf_dataclass import dataclass, function
    >>> @dataclass
    >>> class MyDataclass:
    >>>     ...
    >>> @function
    >>> def my_func(...) -> Tuple[tf.Tensor, MyDataclass]:
    >>>    ...
    >>>    return some_tensor, my_dataclass_instance
    """
    input_type_dict = get_input_type_dict(func)
    output_type = get_output_type(func)
    dictorized_func = pack_function(func, input_type_dict, output_type)
    tf_func = tf.function(func=dictorized_func, **kwargs)
    undictorized_func = unpack_function(tf_func, input_type_dict, output_type)
    return undictorized_func
