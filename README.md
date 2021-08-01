# tf-dataclass

Support python dataclass containers as input and output in callable TensorFlow 2 graph.

## Install
Make sure that ```tensorflow>=2.0.0``` or ```tensorflow-gpu>=2.0.0``` is installed.
```bash
$ pip install tf-dataclass
```

## Why
TensorFlow 2 
[autograph function](https://www.tensorflow.org/api_docs/python/tf/function)
supports only nested structures of python tuples as inputs and output.
(Outputs can be also python dictionaries.)
This is inconvenient once we go beyond small hello world cases,
because we have to work with unstructured armfuls of tensors.
This small package is dedicated to fill this gap by letting 
```@tf.function``` 
decorated functions to take and return pythonic
```dataclass``` 
instancies. 

## Examples of usage:
### 1. Sequential features

```python
import tensorflow as tf
import tf_dataclass

# Batch of sequential features of different length
@tf_dataclass.dataclass
class Sequential:
    feature: tf.Tensor  # shape = [batch, length, channels],    dtype = tf.float32
    length: tf.Tensor   # shape = [batch],                      dtype = tf.int32

# Initialize a batch of two sequences of lengths 6 and 4
input = Sequential(
    feature = tf.random.normal(shape=[2, 6, 3]),
    length = tf.constant([6, 4], dtype=tf.int32),
)
    
# Define a convolution operator with a stride such that length -> length / stride
@tf_dataclass.function
def convolution(input: Sequential, filters: tf.Tensor, stride: int) -> Sequential:
    return Sequential(
        feature = tf.nn.conv1d(input.feature, filters, stride),
        length = tf.math.floordiv(input.length, stride),
    )

# Output is an instance of Sequential with lengths 3 and 2 due to convolution stride = 2
output = convolution(
    input = input,
    filters = tf.random.normal(shape=[1, 3, 7]),
    stride = 2,
)
assert isinstance(output, Sequential)
print(output.length) # -> tf.Tensor([3 2], shape=(2,), dtype=int32)
```

### 2. Minibatch as a data transfer object:
```python
import tensorflow as tf
import tf_dataclass

@tf_dataclass.dataclass
class DataBatch:
    image: tf.Tensor            # shape = [batch, height, width, channels], dtype = tf.flaot32
    label: tf.Tensor            # shape = [batch],                          dtype = tf.int32
    image_file_path: tf.Tensor  # shape = [batch],                          dtype = tf.string
    dataset_name: tf.Tensor     # shape = [batch],                          dtype = tf.string
    ...
    
@tf_dataclass.function
def train_step(input: DataBatch) -> None:
    ...
```

### 3. Containerized outputs:
```python
import tensorflow as tf
import tf_dataclass

@tf_dataclass.dataclass
class ModelOutput:
    loss_value: tf.Tensor   # shape = [batch],  dtype = tf.flaot32
    label: tf.Tensor        # shape = [batch],  dtype = tf.int32
    prediction: tf.Tensor   # shape = [batch],  dtype = tf.int32
    ...
    
    @property
    def mean_loss(self) -> tf.Tensor: # shape = [batch],  dtype = tf.float32
        return tf.reduce_mean(self.loss_value)
    
    @property
    def num_true_predictions(self) -> tf.Tensor: # shape = [batch],  dtype = tf.int32
        return tf.reduce_sum(tf.cast(self.label == self.prediction, dtype=tf.int32))

    @property
    def num_false_predictions(self) -> tf.Tensor: # shape = [batch],  dtype = tf.int32
        return tf.reduce_sum(tf.cast(self.label != self.prediction, dtype=tf.int32))

    ...

@tf_dataclass.function
def get_loss(...) -> ModelOutput:
    ...
```
Such containers can be merged along datasets and workers.

### 4. Easy tensorflow shape and dtype runtime verification:
```python
import tensorflow as tf
import tf_dataclass

@tf_dataclass.dataclass
class Sequential:
    feature: tf.Tensor  # shape = [batch, length, channels], dtype = tf.flaot32
    length: tf.Tensor   # shape = [batch]                    dtype = tf.int32

    def __post_init__(self):
        # Verify feature
        assert self.feature.dtype == tf.float32
        assert len(self.feature.shape) == 3
        
        # Verify length
        assert self.length.dtype == tf.int32
        assert len(self.length.shape) == 1
        
        # Verify batch size
        # Works only in eager mode for better perfomance  
        assert self.feature.shape[0] == self.length.shape[0]

    @property
    def batch_size(self) -> tf.Tensor: # shape = [], dtype = tf.int32
        return tf.shape(self.feature)[0]
```

## Other features:
* Support hierarchical composition.
* Support inheritance including multiple one (for free from original ```dataclass```).
* Highliting, autocomplete, and refactoring from your IDE.

## Usage
1. Import ```dataclass``` and ```function``` from ```tf_dataclass```
```python
from tf_dataclasses import dataclass, function
```
2. <strong>It is mandatory to use return type hints for the function decorated with ```@function```.</strong>
For example,
```python
from typing import Tuple

@dataclass
class MyDataclass:
    ...

@function
def my_func(...) -> Tuple[tf.Tensor, MyDataclass]:
    ...
    return some_tensor, my_dataclass_instance
```
3. Type hints for the arguments are optional but recommended.

4. <strong>Positional arguments are not currently supported</strong>:

For example, for
```python
@function
def my_graph_func(x: ..., y: ...) -> ... :
    ...
```
type
```python
my_graph_func(x=x, y=y)
```
but not
```python
my_graph_func(x, y)
```

## Known Problems
1. IDE autocomplete is currently not well-supported, for example, in PyCharm. 
Solution: use import
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from tf_dataclass import dataclass
```
in each ```*.py``` file where ```dataclass``` is used.

## Under the roof
Dataclasses and their nested structures are simply converted into nested pythonic tuples and back.
This way we wrap given functions such that all inputs and outputs are nested tuples.
Then 
```@tf.function``` 
is applied. Afterward the graph function is wrapped bach to dataclass form.
Type hints are used in python runtime for the graph creation as temples to pack and unpack dataclass arguments. 

## Future plans
1. Support ```tf.cond```, ```tf.case```, ```tf.switch_case```, ```tf.while_loop```, ```tf.Optional```, and 
```tf.data.Iterator```.
2. Support positional arguments.
3. Conversion to ```tf.nest``` structures.