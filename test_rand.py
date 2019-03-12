import tensorflow as tf
tf.set_random_seed(1)

a = tf.random_normal([10], 0, 1, dtype=tf.float32)
b = tf.constant(57)
tf.add
sess = tf.Session()
out = sess.run(a)
print(out)

out = sess.run(a)
print(out)
out = sess.run(a)
print(out)