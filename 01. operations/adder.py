import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

@tf.function
def forward():
    return node1 + node2

print(forward())