import numpy as np
import os
import sys
import pickle
import tflearn
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tflearn.data_utils import to_categorical, shuffle
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, batch_normalization, max_pool_2d
from tflearn.layers.estimator import regression
from PIL import Image
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
    # print(X_train.shape)
    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))/255   #  fileformat pixel_1_r pixel_1_g pixel_1_b
    # print(X_train.shape)
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_train = X_train.astype(np.float32)
    # print(X_train.shape)
    # out = np.dstack((X_train[1][0][:][:], X_train[1][1][:][:], X_train[1][2][:][:]))
    # out = np.array(out)
    # print(out.shape)
    # img = Image.fromarray(np.uint8(X_train[2]))
    # img.show()
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))/255
    X_test = X_test.reshape(-1, 32, 32, 3)
    X_test = X_test.astype(np.float32)

    if one_hot:
        Y_test = to_categorical(Y_test, 10)
        Y_train = to_categorical(Y_train, 10)
        Y_test = Y_test.astype(np.uint8)
        Y_train = Y_train.astype(np.uint8)
    return (X_train, Y_train), (X_test, Y_test)


def convnetwork(preprocess, dataaugmentation):
    network = input_data([None, 32, 32, 3], data_preprocessing=preprocess, data_augmentation=dataaugmentation)
    network = conv_2d(network, nb_filter=32, filter_size=[3, 3], activation='relu')
    network = max_pool_2d(network, kernel_size=[3, 3])
    network = conv_2d(network, nb_filter=64, filter_size=[3, 3], activation='relu')
    network = conv_2d(network, nb_filter=64, filter_size=[3, 3], activation='relu')     # why 2 conv layers?
    network = max_pool_2d(network, kernel_size=[2, 2])
    network = fully_connected(network, n_units=512, activation='relu')      # fully connected with relu?
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy',
                         learning_rate=0.001)

    return network


def main(_):
    # dirname = os.path.join(FLAGS.buckets, "")
    dirname = "data"
    (X_train, Y_train), (X_test, Y_test) = load_data(dirname, one_hot=True)
    X_train, Y_train = shuffle(X_train, Y_train)    # load data

    img_preprocess = ImagePreprocessing()
    img_preprocess.add_featurewise_zero_center()
    img_preprocess.add_featurewise_stdnorm()    # gaussian

    img_augmentation = ImageAugmentation()
    img_augmentation.add_random_flip_leftright()    # data augmentation
    img_augmentation.add_random_rotation()

    network = convnetwork(img_preprocess, img_augmentation)

    model = tflearn.models.DNN(network=network, tensorboard_verbose=0)
    model.fit(X_inputs=X_train, Y_targets=Y_train, n_epoch=20, shuffle=True, show_metric=True,
              validation_set=(X_test, Y_test), batch_size=32, run_id='cifar10_cnn')
    # dirname = os.path.join(FLAGS.checkoutpoint, "model.tfl")
    model_savepath = "data/model.tfl"
    model.save(model_savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--buckets', type=str, default='',
    #                     help='input data path')
    #
    # parser.add_argument('--checkoutpoint', type=str, default='',
    #                     help='output data path')
    # FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main)


# batch_1_data, batch_1_labels = load_batch("data/data_batch_1")
# print(batch_1_data.shape, batch_1_labels)