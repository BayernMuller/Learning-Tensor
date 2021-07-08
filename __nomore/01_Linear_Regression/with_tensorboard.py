import tensorflow as tf

W = tf.Variable(tf.random_normal([1])) # shape is 1 * 1
b = tf.Variable(tf.random_normal([1])) # shape is 1 * 1
x = tf.placeholder(tf.float32)  # domain
# placeholder make you can put value later

lenear_model = W * x + b

y = tf.placeholder(tf.float32) # range

loss = tf.reduce_mean(tf.square(lenear_model - y)) # MSE loss function

tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.01) # set running rate
# I don't know how running rate works yet.
train_step = optimizer.minimize(loss) 
# train to loss function's result keeps lower

x_train = list(range(0, 10)) # real input         [0, 1, 2, ..., 9]
y_train = list(range(0, 30, 3)) # real output     [0, 3, 6, ..., 27]

sess = tf.Session()
sess.run(tf.global_variables_initializer())


merged = tf.summary.merge_all()
tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_train, y: y_train})
    # train with real x and y
    summary = sess.run(merged, feed_dict={x: x_train, y: y_train})
    tensorboard_writer.add_summary(summary, i)

x_test = [4, 8, 10, 17]
# test x values

print(sess.run(lenear_model, feed_dict={x: x_test}))
# print W * x_test + b

sess.close()