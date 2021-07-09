# Variable

### Create Variable

**General**
``` python
import tensorflow as tf
my_variable = tf.Variable(tf.ones([1, 2, 3]))
print(my_variable)
```
``` python
tf.Variable 'Variable:0' shape=(1, 2, 3) dtype=float32, numpy=
array([[[1., 1., 1.],
        [1., 1., 1.]]], dtype=float32)>
```

**In GPU**
```python
with tf.device("/device:GPU:0"):
    v = tf.Variable(tf.zeros([10,10]))
```

**Result of operation**
```python
v = tf.Variable(3)
w = v + 1
print(w)
```
```python
<tf.Tensor: shape=(), dtype=int32, numpy=4>
```

### Operation

```tf.Variable``` + ```Constant``` = ```tf.Tensor```
```python
tf.Variable(1) + 2
```
```python
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

Keep tf.Variable type with ```tf.Variable.assign```, ```tf.Variable.assign_add```
```python
v = tf.Variable(3)
v.assign(10)
print(v)
```
```python
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=10>

# convert Variable to Tensor with "read_value"
print(v.read_value())
<tf.Tensor: shape=(), dtype=int32, numpy=10>
```

### Following Variable

You can make set of Variable with ```tf.Module```
```python
import tensorflow as tf

class MyModule(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(i) for i in range(10)]

class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModule()
        self.v1 = tf.Variable(8.0)

m = MyOtherModule()
print(len(m.variables))
# it prints 12
```