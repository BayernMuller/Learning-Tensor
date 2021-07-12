# Eager Execution

### checkpoint

You can save Tensors to file with checkpoint.

```python
import tensorflow as tf

x = tf.Variable(2.)
checkpoint_path = './pt/'
checkpoint = tf.train.Checkpoint(x = x)
checkpoint.save(checkpoint_path)

x.assign(10.)
print(x) # x is 10.0.

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x) # x is restored. 2.0
```
```
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=10.0>
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
```

### tf.keras.metrics

You can modify the value by giving a new data and get the result with ```tf.keras.metrics.result``` method.

```python
import tensorflow as tf

m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
print(m.result()) # 2.5
m([8, 9])
print(m.result()) # 5.5
```

### user defined gradient

User defined gradient is an easy way to override gradient.

```python
import tensorflow as tf

@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
    t = tf.identity(x)
    def grad_fn(dresult):
        return [tf.clip_by_norm(dresult, norm), None]
    return y, grad_fn
```

### CPU and GPU

```python
import time
import tensorflow as tf

def measure(x, steps):
    # tensorflow initializes GPU on first calculation. exclude on time.
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
      x = tf.matmul(x, x)

    _ = x.numpy()
    # numpy() assures that all calculations are done.
    end = time.time()
    return end - start

shape = (1000, 1000)
steps = 200

# In CPU:
with tf.device("/cpu:0"):
    print("CPU: {} sec".format(measure(tf.random.normal(shape), steps)))

# In GPU:
if tf.config.experimental.list_physical_devices("GPU"):
    with tf.device("/gpu:0"):
        print("GPU: {} sec".format(measure(tf.random.normal(shape), steps)))
else:
    print("No GPU")
```
```
CPU: 6.339999675750732 sec
GPU: 0.14800572395324707 sec
```

The matrix multiply calculation is much faster in GPU than in CPU.

### Copy object to other device

tf.Tensor object can be moved to other device.

```python
import tensorflow as tf

if tf.config.experimental.list_pysical_device("GPU"):
    x = tf.random.normal([10, 10])

    x_gpu0 = x.gpu()
    x_cpu = x.cpu()

    res = tf.matmul(x_cpu, x_cpu) # run on cpu
    res = tf.matmul(x_gpu0, x_gpu0) # run on gpu:0
```