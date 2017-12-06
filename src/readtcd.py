#!/usr/bin/env python3
"""
Convert MIO-TCD format to tfrecord
"""

import os
from pathlib import Path
HOME = str(Path.home())

import tensorflow as tf
import numpy as np
import io
from PIL import Image

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', 'miotcd-train.tfrecord', 'Output folder')
FLAGS = flags.FLAGS
imdir = os.path.join(HOME, 'temp/miotcd-dataset/MIO-TCD-Localization/train/')
annotfilepath = os.path.join(HOME, 'temp/miotcd-dataset/MIO-TCD-Localization/gt_train_train.csv')
classes = ['articulated_truck',
		   'bicycle',
		   'bus',
		   'car',
		   'motorcycle',
		   'motorized_vehicle',
		   'non-motorized_vehicle',
		   'pedestrian',
		   'pickup_truck',
		   'single_unit_truck',
		   'work_van']

def create_tf_example(imdir, curimid, bboxes):
    filename = curimid + '.jpg'
    fullfilename = os.path.join(imdir, filename)
    filename = filename.encode('utf8')

    with tf.gfile.GFile(fullfilename, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = b'jpg'
    #input(bboxes['categids'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(bboxes['xmins']),
        'image/object/bbox/xmax': dataset_util.float_list_feature(bboxes['xmaxs']),
        'image/object/bbox/ymin': dataset_util.float_list_feature(bboxes['ymins']),
        'image/object/bbox/ymax': dataset_util.float_list_feature(bboxes['ymaxs']),
        'image/object/class/text': dataset_util.bytes_list_feature(bboxes['categnames']),
        'image/object/class/label': dataset_util.int64_list_feature(bboxes['categids']),
    }))
    return tf_example


def bboxesinit():
    bboxes = {}
    bboxes['categnames'] = []
    bboxes['xmins'] = []
    bboxes['ymins'] = []
    bboxes['xmaxs'] = []
    bboxes['ymaxs'] = []
    bboxes['categids'] = []
    return bboxes

def stack_features(bboxes, categname, xmin, ymin, xmax, ymax):
    bboxes['categnames'].append(categname.encode('utf8'))
    bboxes['xmins'].append(int(xmin))
    bboxes['ymins'].append(int(ymin))
    bboxes['xmaxs'].append(int(xmax))
    bboxes['ymaxs'].append(int(ymax))
    bboxes['categids'].append(int(classes.index(categname)) + 1)
    return bboxes

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    bboxes = bboxesinit()
    curimid = '-1'

    annotfh = open(annotfilepath)

    for line in annotfh:
        imid, categ, xmin, ymin, xmax, ymax = line.split(',')

        if imid != curimid: # flush bboxes (we assume the imids column is sorted)
            if curimid != '-1':
                tf_example = create_tf_example(imdir, curimid, bboxes)
                writer.write(tf_example.SerializeToString())
                bboxes = bboxesinit()
            curimid = imid

        bboxes = stack_features(bboxes, categ, xmin, ymin, xmax, ymax)

    annotfh.close()
    tf_example = create_tf_example(imdir, curimid, bboxes)
    writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
