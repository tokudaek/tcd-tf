#!/usr/bin/env python3

import numpy as np
import os
import random
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import imghdr

from collections import defaultdict
from io import StringIO
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util


# ## Model preparation 
# In[44]:
HOME = os.environ['HOME']
PATH_TO_CKPT = os.path.join(HOME, 'models/faster_rcnn_resnet50_coco_2017_11_08/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(HOME, 'projects/tensorflow-objdet/research/object_detection/data/mscoco_label_map.pbtxt')

NUM_CLASSES = 2

PATH_TO_TEST_IMAGES_DIR = os.path.join(HOME, 'datasets/20160418_cameriterain_frames/')
numtest = 10
outdir = os.path.join(HOME, 'temp/20171223-resnet50_coco_over_cameriterain/')

if not os.path.exists(outdir): os.mkdir(outdir)

IMAGE_SIZE = (12, 8) # Size, in inches, of the output images.


# In[39]:
detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    print('Model loaded.')


# In[40]:
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[41]:
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[42]:
test_image_paths = []

counter = 0
for f in os.listdir(PATH_TO_TEST_IMAGES_DIR):
    #if counter >= numtest: break
    filename = os.path.join(PATH_TO_TEST_IMAGES_DIR, f)
    if filename.endswith('.jpg'):
        test_image_paths.append(filename)
        counter += 1

print(test_image_paths[:10])


# ## Finally, the detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    for image_path in test_image_paths:
      #if imghdr.what(image_path) != 'jpeg': continue
      #print(imghdr.what(image_path))
      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=6)
    
      plt.imsave(image_path.replace(PATH_TO_TEST_IMAGES_DIR, outdir).replace('jpg', 'png'), image_np)
      print('Saved figure ' + image_path.replace(PATH_TO_TEST_IMAGES_DIR, outdir).replace('jpg', 'png'))

