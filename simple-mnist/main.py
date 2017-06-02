import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import math

# 1 Prepare data
# 2 Inference
# 3 Measure loss
# 4 Optimize net to minimize loss

# Load the MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train)
# create a tf placeholder for the 28 x 28 image
x = tf.placeholder(tf.float32, shape=[None, 784])

# create the label which is y. This is for digits from 0 to 9. 
y_ = tf.placeholder(tf.float32, [None, 10])

# Define the weights and bias values
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
learning_rate = 0.5
steps = 10000
epochs = 10
# Define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Calculate our loss/error rate
cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Add gradient descent
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Init the global vars
init = tf.global_variables_initializer()

# create our tf session
sess = tf.Session()

# run
sess.run(init)

for e in range(epochs):
        
    for i in range(steps):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Test Accuracy:")
    print(test_accuracy)

sess.close()

