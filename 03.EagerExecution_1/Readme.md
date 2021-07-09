# Eager Excution

### What is Eager Excution?
Eager Execution is the programming environment that does not create graph on executing.

```python
import tensorflow as tf
x = [[2.]]
m = tf.matmul(x, x)
print(f'{m}')
```
```
[[4.]]
```

### Broadcasting
```python
import tensorflow as tf
x = [[1,2],[3,4]]
print(tf.add(x, 1))
```
```
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[2, 3],
       [4, 5]], dtype=int32)>
```
tf.add(x, ```1```)

```1``` will be converted to [[1,1],[1,1]] automatically

### Operator Overriding
```python
import tensorflow as tf
a = tf.Variable([[1],[2]])
b = tf.Variable([[3],[4]])
print(a * b)
```
```
tf.Tensor(
[[3]
 [8]], shape=(2, 1), dtype=int32)
```

### Gradient Calcurate
```python
import tensorflow as tf
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w
grad = tape.gradient(loss, w)
print(grad) 
```
```
tf.Tensor([[2.]], shape=(1, 1), dtype=float32)
```

### Variable and Optimizer

Example : [Linear regression](linear_regression.py)
