# Sequential Model

### tf.keras.Sequential

You can make a sequential model by stacking layers in ```tf.keras.Sequential```

Sequential model is proper when there are one input tensor and one output tensor. 

The following cases are not proper using Sequential model.
* There are many inputs or outputs in model.
* There are many inputs or outputs in all layers.
* A layer must be shared.
* It provides nonlinear-topology

**Making Model**
```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(2, activation='relu', name='first'),
    layers.Dense(3, activation='relu', name='second'),
    layers.Dense(4, name='third')
])

x = tf.ones([4, 4])
y = model(x)
print(x)
print(y)
```
```
tf.Tensor(
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]], shape=(4, 4), dtype=float32)
tf.Tensor(
[[-0.7112008  -0.7119014  -0.21034504 -0.12633237]
 [-0.7112008  -0.7119014  -0.21034504 -0.12633237]
 [-0.7112008  -0.7119014  -0.21034504 -0.12633237]
 [-0.7112008  -0.7119014  -0.21034504 -0.12633237]], shape=(4, 4), dtype=float32)
```

Same ways:
```python
model = keras.Sequential()
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(4))
```
```python
first = layers.Dense(2, activation='relu', name='first')
second = layers.Dense(3, activation='relu', name='second')
third = layers.Dense(4, name='third')

x = tf.ones([4, 4,])
y = third(second(first(x)))
```

**Removing layer in model**
```python
print(len(model.layers)) # n
model.pop()
print(len(model.layers)) # n - 1
```

### Sequential.summury()

The method shows shape of ouput in each layers. Useful in debugging.
Following code is example of image classification that shows how summury() works.

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))


model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))

model.summary()
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
module_wrapper_11 (ModuleWra (None, 123, 123, 32)      2432      
_________________________________________________________________
module_wrapper_12 (ModuleWra (None, 121, 121, 32)      9248      
_________________________________________________________________
module_wrapper_13 (ModuleWra (None, 40, 40, 32)        0         
_________________________________________________________________
module_wrapper_14 (ModuleWra (None, 38, 38, 32)        9248      
_________________________________________________________________
module_wrapper_15 (ModuleWra (None, 36, 36, 32)        9248      
_________________________________________________________________
module_wrapper_16 (ModuleWra (None, 12, 12, 32)        0         
_________________________________________________________________
module_wrapper_17 (ModuleWra (None, 10, 10, 32)        9248      
_________________________________________________________________
module_wrapper_18 (ModuleWra (None, 8, 8, 32)          9248      
_________________________________________________________________
module_wrapper_19 (ModuleWra (None, 4, 4, 32)          0         
=================================================================
Total params: 48,672
Trainable params: 48,672
Non-trainable params: 0
_________________________________________________________________
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
module_wrapper_11 (ModuleWra (None, 123, 123, 32)      2432      
_________________________________________________________________
module_wrapper_12 (ModuleWra (None, 121, 121, 32)      9248      
_________________________________________________________________
module_wrapper_13 (ModuleWra (None, 40, 40, 32)        0         
_________________________________________________________________
module_wrapper_14 (ModuleWra (None, 38, 38, 32)        9248      
_________________________________________________________________
module_wrapper_15 (ModuleWra (None, 36, 36, 32)        9248      
_________________________________________________________________
module_wrapper_16 (ModuleWra (None, 12, 12, 32)        0         
_________________________________________________________________
module_wrapper_17 (ModuleWra (None, 10, 10, 32)        9248      
_________________________________________________________________
module_wrapper_18 (ModuleWra (None, 8, 8, 32)          9248      
_________________________________________________________________
module_wrapper_19 (ModuleWra (None, 4, 4, 32)          0         
_________________________________________________________________
module_wrapper_20 (ModuleWra (None, 32)                0         
_________________________________________________________________
module_wrapper_21 (ModuleWra (None, 10)                330       
=================================================================
Total params: 49,002
Trainable params: 49,002
Non-trainable params: 0
_________________________________________________________________
```