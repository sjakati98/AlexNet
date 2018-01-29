from weights_biases import get_weights, get_biases
from layers import conv2d_layer, max_pool_layer, fc_layer

import tensorflow as tf

def alex_net(X, num_classes=10, dropout=0.4, batch_norm=False):

    weights = get_weights(num_classes)
    biases = get_biases(num_classes)

    ## reshape input to be used with network
    X = tf.reshape(X, shape=[-1, 227, 227, 3])
    
    ## model build
    conv1 = conv2d_layer(X, weights['conv1_weights'], biases['conv1_biases'], strides=4, batch_norm=batch_norm)
    pool1 = max_pool_layer(conv1, kernel_size=3, strides=2, padding='SAME')
    conv2 = conv2d_layer(pool1, weights['conv2_weights'], biases['conv2_biases'], strides=1, batch_norm=batch_norm)
    pool2 = max_pool_layer(conv2, kernel_size=3, strides=2, padding='SAME')
    conv3 = conv2d_layer(pool2, weights['conv3_weights'], biases['conv3_biases'], strides=1, batch_norm=batch_norm)
    pool3 = max_pool_layer(conv3, kernel_size=3, strides=2, padding='SAME')
    conv4 = conv2d_layer(pool3, weights['conv4_weights'], biases['conv4_biases'], strides=1, batch_norm=batch_norm)
    pool4 = max_pool_layer(conv4, kernel_size=3, strides=2, padding='SAME')
    conv5 = conv2d_layer(pool4, weights['conv5_weights'], biases['conv5_biases'], strides=1, batch_norm=batch_norm)
    pool5 = max_pool_layer(conv5, kernel_size=3, strides=2, padding='SAME')

    fc1 = fc_layer(pool5, weights['fully_connected1_weights'], biases['fully_connected1_biases'], batch_norm=batch_norm, flatten=True)
    fc2 = fc_layer(fc1, weights['fully_connected2_weights'], biases['fully_connected2_biases'], batch_norm=batch_norm, flatten=False)
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return out
