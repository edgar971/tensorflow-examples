import tensorflow as tf 

sess = tf.Session()

hello = tf.constant("My name is Edgar")

print(sess.run(hello))