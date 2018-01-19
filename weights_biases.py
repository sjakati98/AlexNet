import tensorflow as tf

""" Create the placeholder weights and biases for the network
    in a modular way"""


def get_weights(num_classes=10):
    
    weights = {
        'conv1_weights': tf.Variable(tf.random_normal([11, 11, 3, 96])),
        'conv2_weights': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'conv3_weights': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'conv4_weights': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'conv5_weights': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'fully_connected1_weights': tf.Variable(tf.random_normal([(7**2)*256, 4096])),
        'fully_connected2_weights': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, num_classes]))
    }

    return weights

def get_biases(num_classes=10):

    biases = {
        'conv1_biases': tf.Variable(tf.random_normal([96])),
        'conv2_biases': tf.Variable(tf.random_normal([256])),
        'conv3_biases': tf.Variable(tf.random_normal([384])),
        'conv4_biases': tf.Variable(tf.random_normal([384])),
        'conv5_biases': tf.Variable(tf.random_normal([384])),
        'fully_connected1_biases': tf.Variable(tf.random_normal([4096])),
        'fully_connected2_biases': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    return biases