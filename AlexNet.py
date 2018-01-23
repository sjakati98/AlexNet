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
epochs = 10
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
print("DEBUG: Images loaded")

## TODO: 1. Split images and labels into training and testing sets
##       2. Build network with placeholders
##       3. Train network
##       4. Test network

X_train, y_train = image[:16000], label[:16000]
X_test, y_test = image[16000:], label[16000:]


## create network placeholders
X = tf.placeholder(tf.float32, [-1, 224, 224, 3])
y = tf.placeholder(tf.float32, [-1, num_classes])
dropout_prob = tf.placeholder(tf.float32)

## create the model
model_probs = alex_net(X, num_classes=num_classes, dropout=dropout_prob, batch_norm=batch_norm)
model_probs = tf.reshape(model_probs, [-1, num_classes])

## define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_probs, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

## performance metrics
correct_pred = tf.equal(tf.argmax(model_probs, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

## run the model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

   ## batch training
   for epoch in epochs:
       ## iterate over training examples in specified batch size
        for ## TODO: finish batch training