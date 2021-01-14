import tensorflow as tf

@tf.function
def forward(x, y):
    return x + y

print(forward(10.5, 16.0))