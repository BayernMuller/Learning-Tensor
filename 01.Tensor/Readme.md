# 01. Tensor

### What is "Tensor"

Tensor has following attribute
- Type (such as float32, int32, string)
- shape ([2, 3], [3, 3, 3])

Each eleements in a Tensor have same type.


### Tensor form
``` python
import tensorflow as tf

tf.Variable
tf.constant
tf.placeholder
tf.SparseTensor
# Usual forms of tensor
```
### Rank

number of dimension

sysnonym: order, degree
 
|Rank|Entity|
|--|:---:|
|0|Scalar (Magnitude)|
|1|Vector (Magnitude and Direction)|
|2|Matrix (Table)|
|3|3-Tensor (Cube)|
|n|n-Tensor|

### Rank 0 Tensor

``` python
mammal = tf.Variable("코끼리", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.141234243, tf.float64)
its_complicated = tf.Variable(12.3 - 485j, tf.complex64)
```

### Rank 1 Tensor

``` python
mystr = tf.Variable(["안녕하세요"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)
```