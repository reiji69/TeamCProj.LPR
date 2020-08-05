import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import pandas as pd
import numpy as np

# loading mnist data from tensorflow
from tensorflow.examples.tutorials.mnist import input_data

df = pd.read_csv('A_Z Handwritten Data.csv', header=None)

import matplotlib.pyplot as plt


def showLetters():
    xs = df.sample(16).to_numpy()[:, 1:]
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(xs[i].reshape(28, 28), interpolation='nearest', cmap='Greys')
    plt.show()

showLetters()

# defining the network
input_width = 28
input_height = 28
input_channels = 1
input_pixels = 784

n_conv1 = 32
n_conv2 = 64
stride_conv1 = 1
stride_conv2 = 1
conv1_k = 5
conv2_k = 5
max_pool1_k = 2
max_pool2_k = 2

n_hidden = 1024
n_out = 26

input_size_to_hidden = (input_width // (max_pool1_k * max_pool2_k)) * (
            input_height // (max_pool1_k * max_pool2_k)) * n_conv2

# initialising the weights with random values
weights = {
    "wc1": tf.Variable(tf.random_normal([conv1_k, conv1_k, input_channels, n_conv1])),
    "wc2": tf.Variable(tf.random_normal([conv2_k, conv2_k, n_conv1, n_conv2])),
    "wh1": tf.Variable(tf.random_normal([input_size_to_hidden, n_hidden])),
    "wo": tf.Variable(tf.random_normal([n_hidden, n_out]))
}

biases = {
    "bc1": tf.Variable(tf.random_normal([n_conv1])),
    "bc2": tf.Variable(tf.random_normal([n_conv2])),
    "bh1": tf.Variable(tf.random_normal([n_hidden])),
    "bo": tf.Variable(tf.random_normal([n_out])),
}


# functions for layers
def conv(x, weights, bias, strides=1):
    out = tf.nn.conv2d(x, weights, padding="SAME", strides=[1, strides, strides, 1])
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)
    return out


def maxpooling(x, k=2):
    return tf.nn.max_pool(x, padding="SAME", ksize=[1, k, k, 1], strides=[1, k, k, 1])


# function for forward prop
def cnn(x, weights, biases, keep_prob):
    x = tf.reshape(x, shape=[-1, input_height, input_width, input_channels])
    conv1 = conv(x, weights['wc1'], biases['bc1'], stride_conv1)
    conv1_pool = maxpooling(conv1, max_pool1_k)

    conv2 = conv(conv1_pool, weights['wc2'], biases['bc2'], stride_conv2)
    conv2_pool = maxpooling(conv2, max_pool2_k)

    hidden_input = tf.reshape(conv2_pool, shape=[-1, input_size_to_hidden])
    hidden_output_before_activation = tf.add(tf.matmul(hidden_input, weights['wh1']), biases['bh1'])
    hidden_output_before_dropout = tf.nn.relu(hidden_output_before_activation)
    hidden_output = tf.nn.dropout(hidden_output_before_dropout, keep_prob)

    output = tf.add(tf.matmul(hidden_output, weights['wo']), biases['bo'])
    return output


x = tf.placeholder("float", [None, input_pixels])
y = tf.placeholder(tf.int32, [None, n_out])
keep_prob = tf.placeholder("float")
pred = cnn(x, weights, biases, keep_prob)

# defing cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

# definig optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimize = optimizer.minimize(cost)

# creating session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

accuracy_list = []
average_cost_list = []
# training
batch_size = 1000
for i in range(1000):
    # num_batches = int(372451/batch_size)
    total_cost = 0
    # for j in range(num_batches):
    sample = df.sample(n=batch_size)
    ys = np.zeros((batch_size, 26))
    sample = sample.to_numpy()
    xs = sample[:, 1:]
    raw_y = sample[:, 0]
    ys[np.arange(raw_y.size), raw_y] = 1
    c, _ = sess.run([cost, optimize], feed_dict={x: xs, y: ys, keep_prob: 0.8})
    total_cost += c
    if i != 0 and i % 10 == 0:
        test_batch_number = 500
        sample = df.sample(n=test_batch_number)
        ya = np.zeros((test_batch_number, 26))
        sample = sample.to_numpy()
        xa = sample[:, 1:]
        raw_y = sample[:, 0]
        ya[np.arange(raw_y.size), raw_y] = 1
        cur_accuracy = sess.run(accuracy, feed_dict={x: xa, y: ya, keep_prob: 1.0})
        average_cost_list.append(c)
        accuracy_list.append(cur_accuracy)
        print(f'step {i}, acuracy: {cur_accuracy}')

print('testing----------------------------')
# testing
test_batch_number = 1000
sample = df.sample(n=test_batch_number)
ya = np.zeros((test_batch_number, 26))
sample = sample.to_numpy()
xa = sample[:, 1:]

raw_y = sample[:, 0]
ya[np.arange(raw_y.size), raw_y] = 1
print(f'final accuracy: {sess.run(accuracy, feed_dict={x: xa, y: ya, keep_prob: 1.0})}')

plt.plot(accuracy_list)
plt.ylabel('accuracy')
plt.xlabel('steps')
plt.title('accuracy')
plt.show()

plt.plot(average_cost_list)
plt.ylabel('loss')
plt.xlabel('steps')
plt.title('average Loss')
plt.show()


