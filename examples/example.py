# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model of
Ballé, Laparra, Simoncelli (2017):
End-to-end optimized image compression
https://arxiv.org/abs/1611.01704
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os, random, time
import scipy.misc

def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image

def _load_image():
  """Loads a PNG image file."""
  img_names = os.listdir(args.data_dir)
  img_names = [img_name for img_name in img_names if not img_name == 'Thumbs.db']
  dataset = []
  for img_name in img_names:
    tmp = scipy.misc.imread(args.data_dir + "/" + img_name, mode='RGB')
    tmp = tmp / 255
    dataset.append(tmp) 
  return dataset, img_names

def get_batch(dataset, num_train):
  batch = []
  for i in range(args.batchsize):
    # select image
    idx_img = random.randrange(num_train)
    tmp = dataset[idx_img]
    y, x, _ = tmp.shape
    in_x = random.randint(0, x - args.patchsize)
    in_y = random.randint(0, y - args.patchsize)
    tmp = tmp[in_y:in_y + args.patchsize, in_x:in_x + args.patchsize]  

    # random rotate
    rot_num = random.randint(1, 4)
    tmp = np.rot90(tmp, rot_num)
    
    # random flip left-to-right
    flipflag = random.random() > 0.5
    if flipflag:
        tmp = np.fliplr(tmp) 

    batch.append(tmp) 
  return batch

def save_image(filename, image):
  """Saves an image to a PNG file."""

  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)
    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor

def count_num_trainable_params():
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_num_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_num_params_shape(shape):
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params 

def train():
  """Trains the model."""

  # if args.verbose:
  #   tf.logging.set_verbosity(tf.logging.INFO)

  # # Load all training images into a constant.
  # images = tf.map_fn(
  #     load_image, tf.matching_files(args.data_glob),
  #     dtype=tf.float32, back_prop=False)
  # with tf.Session() as sess:
  #   images = tf.constant(sess.run(images), name="images")

  # # Training inputs are random crops out of the images tensor.
  # crop_shape = (args.batchsize, args.patchsize, args.patchsize, 3)
  # x = tf.random_crop(images, crop_shape)
  # num_pixels = np.prod(crop_shape[:-1])


  crop_shape = (args.batchsize, args.patchsize, args.patchsize, 3)
  x = tf.placeholder(tf.float32, crop_shape)
  num_pixels = np.prod(crop_shape[:-1])

  # Build autoencoder.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, args.num_filters)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_sum(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2 / num_pixels

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.Variable(0, trainable=False, name='global_step')
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  # number of parameters
  num_params = count_num_trainable_params()
  print("num_params: %d" % num_params)

  # For tensorboard
  tf.summary.scalar('loss', train_loss)
  tf.summary.scalar('bpp', train_bpp)
  tf.summary.scalar('mse', train_mse)
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(args.checkpoint_dir+"/logs")
  saver = tf.train.Saver(max_to_keep=100)

  # create tensorflow session
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    file_dir = args.checkpoint_dir + '/results' 
    os.makedirs(file_dir)

    print('Training is started!')
    dataset, img_names = _load_image()

    for _ in range(args.last_step):
      img_batch = get_batch(dataset, len(img_names))
      _, train_summary, loss, global_step = sess.run([train_op, merged, train_loss, step], feed_dict={x: img_batch})

      if global_step % 1000 == 0:
        writer.add_summary(train_summary, global_step=global_step)
        print('step: %d / %d' %(global_step, args.last_step))

      if global_step % 100000 == 0:
        saver.save(sess=sess, save_path=args.checkpoint_dir+"/model.ckpt", global_step=global_step)
        print('Model is saved!')
    print('Training is finished!')


def compress():
  """Compresses an image."""

  # Load input image and add batch dimension.
  # x = load_image(args.input)
  # x = tf.expand_dims(x, 0)
  # x.set_shape([1, None, None, 3])
  x = tf.placeholder(tf.float32, [1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)
  print(x_hat.shape)
  mse = tf.reduce_sum(tf.squared_difference(x * 255, x_hat)) / num_pixels

  with tf.Session() as sess:
    # Load the latest model checkpoint and test images. 
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    dataset, img_names = _load_image()

    for img, img_name in zip(dataset, img_names):
      # Get the compressed string and the tensor shapes.
      _string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)], feed_dict={x: [img]})

      # Write a binary file with the shape information and the compressed string.
      file_name = args.checkpoint_dir + '/results/' + img_name[:-4] + '.bin'
      with open(file_name, "wb") as f:

      # with open(args.output, "wb") as f:
        f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(_string)

      # If requested, transform the quantized image back and measure performance.
      if args.verbose:
        # To print the results, the size of images must be a multiple of 16. 
        # eval_bpp, mse, num_pixels = sess.run([eval_bpp, mse, num_pixels], feed_dict={x: [img]})
        _eval_bpp, _num_pixels = sess.run([eval_bpp, num_pixels], feed_dict={x: [img]})

        # The actual bits per pixel including overhead.
        bpp = (8 + len(_string)) * 8 / _num_pixels

        # print("Mean squared error: {:0.4}".format(mse))
        print("Information content of this image in bpp: {:0.4}".format(_eval_bpp))
        print("Actual bits per pixel for this image: {:0.4}".format(bpp))

def decompress(file_name, out_name):
  """Decompresses an image."""
  # Read the shape information and compressed string from the binary file.
  with open(file_name, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [args.num_filters]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(
      strings, y_shape, channels=args.num_filters)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(out_name, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)

    # n = 1
    # start = time.time()
    # for _ in range(n):
    #   sess.run(op)
    # end = time.time()
    # print('time: ', (end - start)/n)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress"],
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options.")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--verbose", "-v", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--data_dir", default="D:/Dataset/Webtoon/train_sound",
      help="Directory of training dataset")
  parser.add_argument(
      "--da", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=128,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.1, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  parser.add_argument(
    "--device", default='0',
    help="Select GPU device.")

  args = parser.parse_args()

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=args.device

  if args.command == "train":
    train()
  elif args.command == "compress":
    # if args.input is None or args.output is None:
    #   raise ValueError("Need input and output filename for compression.")
    compress()
  elif args.command == "decompress":
    # if args.input is None or args.output is None:
    #   raise ValueError("Need input and output filename for decompression.")
    
    _, img_names = _load_image()
    for img_name in img_names: 
      file_name = args.checkpoint_dir + '/results/' + img_name[:-4] + '.bin'
      out_name = args.checkpoint_dir + '/results/' + img_name
      decompress(file_name, out_name)
      tf.reset_default_graph()
