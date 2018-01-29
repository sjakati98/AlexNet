import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf

CLASSES = [
    'person',
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

def one_hot_image_arrays(ROOT_DIR='/Volumes/Shishir128/Datasets/VOCdevkit/VOC2012/'):

    ## takes in a path to the directory in which VOC is held
    IMG_DIR = os.path.join(ROOT_DIR, 'JPEGImages')
    ANN_DIR = os.path.join(ROOT_DIR, 'Annotations')
    image_arrays = []
    ## every image annotation file
    for filename in os.listdir(ANN_DIR):
        ## skip hidden files
        if filename.startswith('.'): continue
        ## parse xml to extract the classes in the image
        image_array = np.zeros(len(CLASSES), dtype=int)
        ## load xml tree
        tree = ET.parse(os.path.join(ANN_DIR, filename))
        
        ## get objects in tree
        for class_ in tree.getroot().iter('object'):
            image_array[CLASSES.index(class_[0].text)] = 1

        ## turn filename into corresponding jpeg name
        filename_jpeg = filename.replace('xml', 'jpeg')
        filename_jpeg = os.path.join(IMG_DIR, filename_jpeg)
        image_arrays.append([filename_jpeg, image_array])
    
    
    return image_arrays


def load_images(images):
    label = images[1]
    image = tf.read_file(images[0])
    image = tf.image.decode_jpeg(image, channels=3)
    return image, label

    

def image_to_tensor(image_list):
    filename_queue = tf.train.string_input_producer(image_list)
    image_reader = tf.WholeFileReader()
    key, image_contents = image_reader.read(filename_queue)
    images = tf.image.decode_jpeg(image_contents, channels=3)
    print("DEBUG:", images.shape)
    float_images = tf.cast(images, tf.float32)
    return float_images
