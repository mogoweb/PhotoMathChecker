import keras
from keras.datasets import mnist
import tensorflow as tf
from research.object_detection.utils import dataset_util

import numpy as np
from PIL import Image, ImageOps
import os
import random

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def convert(size, box):
  dw = 1./size[0]
  dh = 1./size[1]
  x = (box[0] + box[1])/2.0
  y = (box[2] + box[3])/2.0
  w = box[1] - box[0]
  h = box[3] - box[2]
  x = x*dw
  w = w*dw
  y = y*dh
  h = h*dh
  return (x,y,w,h)

def save_image(filename, data_array):

  #bgcolor = (0xff, 0xff, 0xff)
  bgcolor = (0x00, 0x00, 0xff)
  screen = (500, 375)

  img = Image.new('RGB', screen, bgcolor)

  mnist_img = Image.fromarray(data_array.astype('uint8'))
  mnist_img_invert = ImageOps.invert(mnist_img)

  #w = int(round(mnist_img.width * random.uniform(8.0, 10.0)))
  w = int(mnist_img.width*10)
  mnist_img_invert = mnist_img_invert.resize((w,w))

  #x = random.randint(0, img.width-w)
  #y = random.randint(0, img.height-w)
  x = int((img.width-w)/2)
  y = int((img.height-w)/2)
  img.paste(mnist_img_invert, (x, y))
  img.save(filename)

  return convert((img.width,img.height), (float(x), float(x+w), float(y), float(y+w)))

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

DIR_NAME = "JPEGImages"
if not os.path.exists(DIR_NAME):
  os.mkdir(DIR_NAME)

LABEL_DIR_NAME = "labels"
if not os.path.exists(LABEL_DIR_NAME):
  os.mkdir(LABEL_DIR_NAME)


def create_mnist_tf_example():
  """Creates a tf.Example proto from sample cat image.

  Args:
      encoded_cat_image_data: The jpg encoded data of the cat image.

  Returns:
      example: The created tf.Example.
  """

  j = 0
  no = 0

  for li in [x_train, x_test]:
    j += 1
    i = 0
    print("[---------------------------------------------------------------]")
    for x in li:
      # Write Image file
      filename = "{0}/{1:05d}.jpg".format(DIR_NAME, no)
      print(filename)
      ret = save_image(filename, x)
      print(ret)

      # Write label file
      label_filename = "{0}/{1:05d}.txt".format(LABEL_DIR_NAME, no)
      print(label_filename)
      f = open(label_filename, 'w')

      y = 0
      if j == 1:
        y = y_train[i]
      else:
        y = y_test[i]

      str = "{0:d} {1:f} {2:f} {3:f} {4:f}".format(y, ret[0], ret[1], ret[2], ret[3])
      f.write(str)
      f.close()

      height = 500
      width = 375
      image_format = b'jpg'

      xmins = ret[0]
      xmaxs = ret[1]
      ymins = ret[2]
      ymaxs = ret[3]
      classes_text = ['Cat']
      classes = [y]

      tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(),
          'image/format': dataset_util.bytes_feature(image_format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))

      i += 1
      no += 1


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()