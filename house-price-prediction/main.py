import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate house data
num_house = 500
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate house prices
np.random.seed(42)

house_price = house_size * 100.0 + np.random.randint(low=2000, high=70000, size=num_house)


# Normalize the data
def normalize(array):
    return (array - array.mean()) / array.std()

# Trainig data sets
num_train_samples = math.floor(num_house * 0.7)

train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Test data sets
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Define tensors
tf_house_size = tf.placeholder("float", name="house_size")
tf_house_price = tf.placeholder("float", name="price")

# Define vars
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")


# Define price prediction
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# Loss error rate
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_house_price, 2))/(2*num_train_samples)
learning_rate = 0.9

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    epochs = 100

    for epoch in range(epochs):
        # Fit all training data
        for (x, y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_house_price: y})
        
        if(epoch + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
            print("iteration #:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    
    print("Finished Optimizing")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), "\n")

