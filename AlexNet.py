import os
import sys

from network import alex_net
from load_pascal import one_hot_image_arrays, load_images
import tensorflow as tf
import numpy as np




""" 1. Import the images from the directory
    2. Split the images into training and testing data
    3. Train the model on the training data
    4. Test the model
"""

num_classes = 20
learning_rate = 1e-4
batch_size = 64
dropout = 0.4
batch_norm = True

input_data = one_hot_image_arrays(ROOT_DIR='/Volumes/Shishir128/Datasets/VOCdevkit/VOC2012/')
images_list = np.array([x[0] for x in input_data])
labels_list = np.array([x[1] for x in input_data])

print("DEGUB: Images list size:", images_list.shape)
print("DEGUB: Labels list size:", labels_list.shape)

## convert lists to tensors
images = tf.convert_to_tensor(images_list, tf.string)
labels = tf.convert_to_tensor(labels_list, tf.float32)
print("DEBUG: Lists converted to tensors")


image_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
image, label = load_images(image_queue) # fix the path joiner for tensors
print("Images loaded")

## TODO: 1. Split images and labels into training and testing sets
##       2. Build network with placeholders
##       3. Train network
##       4. Test network

