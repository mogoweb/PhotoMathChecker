import keras
from keras.datasets import mnist
import tensorflow as tf
from object_detection.utils import dataset_util

import numpy as np
from PIL import Image, ImageOps
import os
import random

flags = tf.app.flags
flags.DEFINE_string('train_tf_output_path', '../data/mnist_train.record', 'Path to train output TFRecord')
flags.DEFINE_string('test_tf_output_path', '../data/mnist_test.record', 'Path to test output TFRecord')
flags.DEFINE_string('train_jpg_output_path', '../JPEGImages/train', 'Path to output mnist train jpegs')
flags.DEFINE_string('test_jpg_output_path', '../JPEGImages/test', 'Path to output mnist test jpegs')
FLAGS = flags.FLAGS

IMAGE_WIDTH = 50
IMAGE_HEIGHT = 38


def save_image(filename, data_array):

  bgcolor = (0xff, 0xff, 0xff)
  screen = (IMAGE_WIDTH, IMAGE_HEIGHT)

  img = Image.new('RGB', screen, bgcolor)

  mnist_img = Image.fromarray(data_array.astype('uint8'))
  mnist_img_invert = ImageOps.invert(mnist_img)

  w = int(mnist_img.width)
  mnist_img_invert = mnist_img_invert.resize((w, w))

  x = int((img.width - w) / 2)
  y = int((img.height - w) / 2)
  img.paste(mnist_img_invert, (x, y))
  img.save(filename)

  return float(x) / img.width, float(x+w) / img.width, float(y) / img.height, float(y+w) / img.height


def create_mnist_tf_examples(X, Y, jpg_output_path):
  """Creates a tf.Example proto from mnist dataset.

  Args:
      X: mnist dataset.
      Y: mnist label array.

  Returns:
      example: The created tf.Example.
  """

  mnist_tf_examples = []

  i = 0
  for x in X:

    # Write Image file
    filename = "{0}/{1:05d}.jpg".format(jpg_output_path, i)
    print(filename)
    ret = save_image(filename, x)
    print(ret)

    with tf.gfile.GFile(filename, 'rb') as fid:
      encoded_jpg = fid.read()

    y = Y[i]

    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    image_format = b'jpg'

    xmins = [ret[0]]
    xmaxs = [ret[1]]
    ymins = [ret[2]]
    ymaxs = [ret[3]]
    print("y:", y)
    classes_text = [str(y).encode('utf8')]
    classes = [y + 1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    i += 1
    mnist_tf_examples.append(tf_example)

  return mnist_tf_examples


def main(_):
  if not os.path.exists(FLAGS.train_jpg_output_path):
    os.makedirs(FLAGS.train_jpg_output_path)

  if not os.path.exists(FLAGS.test_jpg_output_path):
    os.makedirs(FLAGS.test_jpg_output_path)

  output_dir = os.path.dirname(FLAGS.train_tf_output_path)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # the data, shuffled and split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  mnist_tf_examples = create_mnist_tf_examples(x_train, y_train, FLAGS.train_jpg_output_path)
  writer = tf.python_io.TFRecordWriter(FLAGS.train_tf_output_path)
  for example in mnist_tf_examples:
    writer.write(example.SerializeToString())
  writer.close()

  mnist_tf_examples = create_mnist_tf_examples(x_test, y_test, FLAGS.test_jpg_output_path)
  writer = tf.python_io.TFRecordWriter(FLAGS.test_tf_output_path)
  for example in mnist_tf_examples:
    writer.write(example.SerializeToString())
  writer.close()


if __name__ == '__main__':
  tf.app.run()