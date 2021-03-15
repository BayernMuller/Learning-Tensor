import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder = a + b
# adder = tf.add(a, b)

sess = tf.Session()
print(sess.run(adder, feed_dict={a:3, b:4.5}))
print(sess.run(adder, feed_dict={a:[1, 5], b:[6, 7]}))