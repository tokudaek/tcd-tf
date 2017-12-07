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
import argparse

from object_detection.utils import dataset_util


flags = tf.app.flags
FLAGS = flags.FLAGS

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

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(bboxes['xmins']/width),
        'image/object/bbox/xmax': dataset_util.float_list_feature(bboxes['xmaxs']/width),
        'image/object/bbox/ymin': dataset_util.float_list_feature(bboxes['ymins']/height),
        'image/object/bbox/ymax': dataset_util.float_list_feature(bboxes['ymaxs']/height),
        'image/object/class/text': dataset_util.bytes_list_feature(bboxes['categnames']),
        'image/object/class/label': dataset_util.int64_list_feature(bboxes['categids']),
    }))
    return tf_example


def bboxesinit():
    bboxes = {}
    bboxes['categnames'] = []
    bboxes['xmins'] = np.array([])
    bboxes['ymins'] = np.array([])
    bboxes['xmaxs'] = np.array([])
    bboxes['ymaxs'] = np.array([])
    bboxes['categids'] = []
    return bboxes

def stack_features(bboxes, categname, xmin, ymin, xmax, ymax):
    bboxes['categnames'].append(categname.encode('utf8'))
    bboxes['xmins'] = np.append(bboxes['xmins'], float(xmin))
    bboxes['ymins'] = np.append(bboxes['ymins'], float(ymin))
    bboxes['xmaxs'] = np.append(bboxes['xmaxs'], float(xmax))
    bboxes['ymaxs'] = np.append(bboxes['ymaxs'], float(ymax))
    bboxes['categids'].append(int(classes.index(categname)) + 1)
    return bboxes

def args_ok(args):
    if not args.input or not args.output or not args.imdir:
        print('Missing arguments --input --imdir --outdir')
        return False
    valid = True
    if not os.path.exists(args.input):
        print('Provided CSV "{}" does not exist'.format(args.input))
        valid = False
    if not os.path.exists(args.imdir):
        print('Provided images dir "{}" does not exist'.format(args.imdir))
        valid = False
    if os.path.exists(args.output):
        print('Provided output file "{}" already exists.'.format(args.output))
        ans = input('Do you want to overwrite it? ')
        if ans not in ['Y', 'y']: valid = False
    return valid

def main(argv):
    parser = argparse.ArgumentParser(description='Generate the tfrecord based on tcd csv format.')
    parser.add_argument('--input', help='Input CSV output in TCD format')
    parser.add_argument('--imdir', help='Images directory')
    parser.add_argument('--output', help='Output TFrecord output')
    args = parser.parse_args()
    
    if not args_ok(args): return
    writer = tf.python_io.TFRecordWriter(args.output)

    bboxes = bboxesinit()
    curimid = '-1'

    annotfh = open(args.input)

    for line in annotfh:
        imid, categ, xmin, ymin, xmax, ymax = line.split(',')

        if imid != curimid: # flush bboxes (we assume the imids column is sorted)
            if curimid != '-1':
                tf_example = create_tf_example(args.imdir, curimid, bboxes)
                writer.write(tf_example.SerializeToString())
                bboxes = bboxesinit()
            curimid = imid

        bboxes = stack_features(bboxes, categ, xmin, ymin, xmax, ymax)

    annotfh.close()
    tf_example = create_tf_example(args.imdir, curimid, bboxes)
    writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()
