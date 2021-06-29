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
import inspect
from typing import Callable, Dict


def get_output_type(func) -> type:
    """
    Returns output type for given function
    Example:
        >>> def func(x: int, y: MyBetterDataclass) -> MyBetterDataclass2:
        >>>     ...
        >>> assert get_output_type(func) == MyBetterDataclass2

    @param func: Callable[[...], Any]
    @return: type of the output
    """
    output_type = inspect.getfullargspec(func).annotations.get("return")
    if not output_type:
        raise ValueError(f"Return type for function {func} is not determined. "
                         f"Probably type hints for the return type are not specified.")
    if isinstance(output_type, str):
        output_type = eval(output_type)

    return output_type


def get_input_type_dict(func: Callable) -> Dict[str, type]:
    """
    Returns a dict like {argname: type} for given function inputs
    Examples:
        >>> @dataclass
        >>> class Base:
        >>>     ...
        >>> @dataclass
        >>> class Base2:
        >>>     ...
        >>> def func(x: int, y: Base) -> Base2:
        >>>     ...
        >>> assert get_output_type(func) == {"x": int, "y": Base}

    @param func: Callable[[...], Any]
    @return: dict like {argname: type}
    """
    return {
        arg_name: inspect.getfullargspec(func).annotations.get(arg_name, object)
        for arg_name in inspect.getfullargspec(func).args
    }