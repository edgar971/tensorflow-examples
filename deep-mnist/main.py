import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import math

# Load dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# define placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])

# change the input image to a 28 x 28 x 1 vector
x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

# define functions to create weights, baises, convolutions, and pooling layers. 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape): 
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Define layers
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# Do convolution on images
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# run through max pool layer
h_pool1 = max_pool_2x2(h_conv1)

# 2nd conv layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# connected output pooling layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Implement dropout to reduce overfitting
keep_prop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define model 
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Measure loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_label))

# loss optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# What is correct
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_label, 1))

# How accurate is it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Init our vars
sess.run(tf.global_variables_initializer())

# train the model

import time

# steps and epochs
steps = 100
epochs = 15

start_time = time.time()
end_time = time.time()

for e in range(epochs):
    print("working")
    for i in range(steps):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_label: batch[1], keep_prop: 0.5})

    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_label: batch[1], keep_prop: 1.0})
    end_time = time.time()
    print("epoch {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(e, end_time-start_time, train_accuracy*100))


sess.close()