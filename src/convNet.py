import numpy as np
import tensorflow as tf


def conv(bottom, filt_name, filt_shape):
    filt = tf.Variable(initial_value=tf.truncated_normal(shape=filt_shape, stddev=0.01), name=filt_name)
    bias = tf.Variable(initial_value=tf.truncated_normal(shape=[filt_shape[3], ], stddev=0.001),
                       name=filt_name + "bias")
    layer = tf.nn.conv2d(bottom, filt, strides=[1, 1, 1, 1], padding="SAME") + bias
    return tf.nn.relu(layer)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def f_c(bottom, w_name, w_shape,activation_func=True):
    w_fc = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name=w_name)
    bias_fc = tf.Variable(tf.constant(0.1, shape=[w_shape[1], ]), name=w_name + "bias")
    if activation_func:
        return tf.nn.relu(tf.matmul(bottom, w_fc) + bias_fc)
    else:
        return tf.matmul(bottom, w_fc) + bias_fc


def net(input_image, keep_prob):
    
    input_image = tf.reshape(input_image, [-1, 28, 28, 1])

    conv_5x5_in = conv(input_image, "conv_5x5_in", [5, 5, 1, 64])

    conv_2 = conv(conv_5x5_in, "conv_2", [3, 3, 64, 64])
    pool1 = max_pool(conv_2, 'pool1')

    conv_3 = conv(pool1, "conv_3", [3, 3, 64, 64])
    pool2 = max_pool(conv_3, 'pool2')

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

    fc1 = f_c(pool2_flat, w_name="fc_1", w_shape=[7 * 7 * 64, 1024])
    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

    fc2 = f_c(h_fc1_drop, w_name="fc_2", w_shape=[1024, 10],activation_func=False)

    prediction = tf.argmax(fc2,1)
    
    return prediction
