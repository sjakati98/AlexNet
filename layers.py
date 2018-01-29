import tensorflow as tf

def conv2d_layer(bottom, weights, biases, strides=2, batch_norm=False, padding='SAME'):

    ## bottom: rank-3 tensor with shape [batch, height, width, channels]
    ## weights: shared convolution weights
    ## biases: added bias vales

    preactivated = tf.nn.conv2d(bottom, weights, [1,strides, strides, 1], padding=padding)
    if batch_norm:
        preactivated = tf.layers.batch_normalization(preactivated)
    else:
        preactivated = tf.nn.bias_add(preactivated, biases)
    activated = tf.nn.relu(preactivated)
    return activated

def max_pool_layer(bottom, kernel_size=3, strides=2, padding='SAME'):

    pool_out = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1], padding=padding)
    return pool_out
    

def fc_layer(bottom, weights, biases, batch_norm=False, flatten=False, dropout=0.4):

    ## input coming from conv layers, so it must be flattened
    if flatten:
        bottom = tf.reshape(bottom, [-1, weights.get_shape().as_list()[0]])
    fully_connected = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
    
    activated = tf.nn.relu(fully_connected)
    activated = tf.nn.dropout(fully_connected, keep_prob=dropout)
    return activated

    