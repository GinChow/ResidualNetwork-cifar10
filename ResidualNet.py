from __future__ import division, print_function, absolute_import
import numpy as np
import os
import sys
import pickle
import tflearn
import tensorflow as tf

from tensorflow.python.lib.io import file_io
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.layers import input_data, conv_2d, fully_connected, batch_normalization, max_pool_2d, avg_pool_2d, residual_block
from tflearn.layers.estimator import regression

import argparse

FLAGS = None


def load_batch(path):
    obj = file_io.read_file_to_string(path, binary_mode=True)
    if sys.version_info > (3, 0):
        d = pickle.loads(obj, encoding='latin1')
    else:
        d = pickle.loads(obj)
    data = np.array(d["data"])
    labels = np.array(d["labels"])
    return data, labels


def load_data(dirname, one_hot=False):
    X_train = []
    Y_train = []

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate((X_train, data))
            Y_train = np.concatenate((Y_train, labels))

    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_batch(fpath)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))/255
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_train = X_train.astype(np.float32)

    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))/255
    X_test = X_test.reshape(-1, 32, 32, 3)
    X_test = X_test.astype(np.float32)

    if one_hot:
        Y_train = to_categorical(Y_train, 10).astype(np.uint8)
        Y_test = to_categorical(Y_test, 10).astype(np.uint8)
    return (X_train, Y_train), (X_test, Y_test)


def ResNet(img_pre, img_aug):
    resNet = input_data(shape=[None, 32, 32, 3], data_augmentation=img_aug, data_preprocessing=img_pre)
    resNet = tf.layers.conv2d(resNet, filters=8, kernel_size=3, padding='valid')
    resNet = tf.layers.batch_normalization(resNet)
    resNet = tf.nn.relu(resNet)

    for i in range(2):
        resNet = Resblock(resNet, filters=16, kernelsize=3)
    #  downsampling
    resNet = tf.layers.max_pooling2d(resNet, pool_size=2, strides=2, padding='same')  # downsamplewith conv?

    resNet = Residentityblock(resNet, filters=16)
    resNet = Residentityblock(resNet, filters=16)

    for i in range(2):
        resNet = Resblock(resNet, filters=32, kernelsize=3)
    # downsampleing
    resNet = tf.layers.max_pooling2d(resNet, pool_size=2, strides=2, padding='same')

    resNet = Residentityblock(resNet, filters=32)
    resNet = Residentityblock(resNet, filters=32)

    for i in range(2):
        resNet = Resblock(resNet, filters=64, kernelsize=3)
    resNet = Residentityblock(resNet, filters=64)
    resNet = Residentityblock(resNet, filters=64)
    resNet = Residentityblock(resNet, filters=64)
    resNet = Residentityblock(resNet, filters=64)

    resNet = tf.layers.average_pooling2d(resNet, pool_size=2, strides=(2, 2), padding='same')  # tensor size 4*4*64

    resNet = fully_connected(resNet, n_units=10, activation='softmax')
    resNet = regression(resNet, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.1)

    return resNet






def Resblock(input_x, filters, kernelsize):
    x_origin = tf.layers.conv2d(input_x, filters=filters, kernel_size=[1, 1], padding='valid')
    x_origin = tf.layers.batch_normalization(x_origin)

    conv_1_output = tf.layers.conv2d(input_x, filters=filters, kernel_size=[kernelsize, kernelsize], padding='same')
    bn_1_output = tf.layers.batch_normalization(conv_1_output)
    relu_1_output = tf.nn.relu(bn_1_output)
    conv_2_output = tf.layers.conv2d(relu_1_output, filters=filters, kernel_size=[kernelsize, kernelsize], padding='same')
    #  shortcut
    bn_2_output = tf.layers.batch_normalization(conv_2_output) + x_origin
    output = tf.nn.relu(bn_2_output)
    return output


def Residentityblock(input_x, filters):
    x_origin = input_x

    conv_1_output = tf.layers.conv2d(input_x, filters=filters/2, kernel_size=3, padding='same')
    bn_1_output = tf.layers.batch_normalization(conv_1_output)
    relu_1_output = tf.nn.relu(bn_1_output)

    conv_2_output = tf.layers.conv2d(relu_1_output, filters=filters/2, kernel_size=3, padding='same')
    bn_2_output = tf.layers.batch_normalization(conv_2_output)
    relu_2_output = tf.nn.relu(bn_2_output)

    conv_3_output = tf.layers.conv2d(relu_2_output, filters=filters, kernel_size=3, padding='same')
    #   shortcut
    bn_3_output = tf.layers.batch_normalization(conv_3_output + x_origin)
    output = tf.nn.relu(bn_3_output)

    return output



def main(_):
    dirname = os.path.join(FLAGS.buckets, "")
    (X_train, Y_train), (X_test, Y_test) = load_data(dirname, one_hot=True)
    X_train, Y_train = shuffle(X_train, Y_train)

    img_preprocessing = ImagePreprocessing()
    img_preprocessing.add_featurewise_zero_center()
    img_preprocessing.add_featurewise_stdnorm()

    img_augmentation = ImageAugmentation()
    img_augmentation.add_random_flip_leftright()
    img_augmentation.add_random_rotation()

    resnet = ResNet(img_preprocessing, img_augmentation)

    model = tflearn.models.DNN(network=resnet, tensorboard_verbose=0)
    model.fit(X_inputs=X_train, Y_targets=Y_train, n_epoch=100, shuffle=True, show_metric=True,
              validation_set=(X_test, Y_test), batch_size=128, run_id='resnet-cifar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')

    parser.add_argument('--checkoutpoint', type=str, default='',
                        help='output data path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)

