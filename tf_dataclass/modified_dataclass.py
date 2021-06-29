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
import dataclasses as dataclasses_orig

from typing import Dict, Any, Union, Tuple, get_type_hints


@classmethod
def from_dict_to_tf_dataobject(cls, data_dict: Dict[str, Any]):
    for name, type_val in get_type_hints(cls).items():
        if is_dataclass(type_val):
            sub_dataclass_dict = data_dict.get(name)
            assert sub_dataclass_dict
            assert isinstance(sub_dataclass_dict, dict)
            inner_dataobject = type_val.from_dict(data_dict=sub_dataclass_dict)
            data_dict[name] = inner_dataobject

    return cls(**data_dict)


@property
def tf_dataobject_to_dict(self) -> Dict[str, Any]:
    data_dict = {}
    for name, value in self.__dict__.items():
        if is_dataclass(value):
            sub_dataclass_dict = value.as_dict
        else:
            sub_dataclass_dict = value
        data_dict[name] = sub_dataclass_dict
    return data_dict


@property
def tf_to_tuple(self) -> Tuple:
    output_list = []
    for name, value in self.__dict__.items():
        if is_dataclass(value):
            sub_dataclass_tuple = value.as_tuple
        else:
            sub_dataclass_tuple = value
        output_list.append(sub_dataclass_tuple)
    return tuple(output_list)


@classmethod
def from_tuple_to_tf_dataobject(cls, data_tuple: Tuple):
    data_dict = {}
    for field, (name, type_val) in zip(data_tuple, get_type_hints(cls).items()):
        if is_dataclass(type_val):
            sub_data_tuple = field
            assert sub_data_tuple
            assert isinstance(sub_data_tuple, tuple)
            inner_dataobject = type_val.from_tuple(data_tuple=sub_data_tuple)
            data_dict[name] = inner_dataobject
        else:
            data_dict[name] = field

    return cls(**data_dict)


def dataclass(
        _cls = None,
        *arg,
        init = True,
        repr = True,
        eq = True,
        order = False,
        unsafe_hash = False,
):
    """
    Modification of dataclasses.dataclass that adds the following methods:
        1. dataclass.from_tuple(data_tuple: Tuple[Any])
        2. dataclass.from_dict(data_dict: Dict[str, Any])
    and properties:
        1. dataclass.as_tuple
            return a tuple version of the dataclass such that all internal subdataclassed
            (modified with the current decorator) are also unpacked to dicts
        2. dataclass.as_dict -> Tuple
            return a dict version of the dataclass such that all internal subdataclassed
            (modified with the current decorator) are also unpacked to dicts
        3. is_better_dataclass equal to True if the current instance is actually a modified dataclass
    """
    _cls.from_dict = from_dict_to_tf_dataobject
    _cls.as_dict = tf_dataobject_to_dict
    _cls.from_tuple = from_tuple_to_tf_dataobject
    _cls.as_tuple = tf_to_tuple
    _cls.is_better_dataclass = True

    return dataclasses_orig.dataclass(
        _cls=_cls,
        *arg,
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=True,
    )


def is_dataclass(class_or_instance: Union[type, Any]) -> bool:
    """
    @param class_or_instance:
    @return: True is @param class_or_instance is an instance or type of tf_dataclass.dataclass
    """
    cls = class_or_instance if isinstance(class_or_instance, type) else type(class_or_instance)
    return dataclasses_orig.is_dataclass(cls) and hasattr(cls, "is_better_dataclass") and cls.is_better_dataclass