import os
import sys

from network import alex_net
from load_pascal import one_hot_image_arrays, load_images, image_to_tensor
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


images = image_to_tensor(tf.convert_to_tensor(images_list, tf.string))
labels = tf.convert_to_tensor(labels_list, tf.float32)



X_train, y_train = images[:16000], labels[:16000]
X_test, y_test = images[16000:], labels[16000:]

print("DEBUG:", X_train.shape)

## create network placeholders
X = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
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

    print(X_train.shape)
    train_image_batch, train_label_batch = tf.train.batch([X_train, y_train], batch_size=batch_size)
    test_image_batch, test_label_batch = tf.train.batch([X_test, y_test], batch_size=batch_size)

   ## batch training
    for epoch in epochs:
       ## iterate over training examples in specified batch size
        iter_ = 0
        while (iter_ * batch_size) < 16000:
            batch_x, batch_y = sess.run([train_image_batch, train_label_batch])
            sess.run(optimizer, feed_dict={
               X: batch_x,
               y:batch_y,
               dropout_prob:0.5
            })
            los, acc = sess.run([loss, accuracy], feed_dict={
                X: batch_x,
                y: batch_y,
                dropout_prob: 1
            })
            print("Minibatch loss:", los, "Minibatch accuracy:", acc)
            iter_ += 1
    
        print("Epoch complete!")
    
    print("DEBUG: Training Complete")

    ## testing
    test_x, test_y = sess.run([test_image_batch, test_label_batch])
    accuracy = sess.run(optimizer, feed_dict={X: test_x, y:test_y, dropout_prob:1})
    print("Accuracy:", accuracy)