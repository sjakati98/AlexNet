import os
import xml.etree.ElementTree as ET

import numpy as np


ROOT_DIR = '/Volumes/Shishir128/Datasets/VOCdevkit/VOC2012/'
IMG_DIR = os.path.join(ROOT_DIR, 'JPEGImages')
ANN_DIR = os.path.join(ROOT_DIR, 'Annotations')

CLASSES = [
    'person',
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

def one_hot_image_arrays():
    image_arrays = []
    ## every image annotation file
    for filename in os.listdir(ANN_DIR):
        if filename.startswith('.'): continue
        ## parse xml to extract the classes in the image
        image_array = np.zeros(len(CLASSES), dtype=int)
        ## load xml tree
        tree = ET.parse(os.path.join(ANN_DIR, filename))
        
        ## get objects in tree
        for class_ in tree.getroot().iter('object'):
            image_array[CLASSES.index(class_[0].text)] = 1
    
        image_arrays.append(image_array)
        print(image_array)
    
    
    return image_arrays



