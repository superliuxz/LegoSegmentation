import cv2
import logging
import os
from PIL import Image
import tarfile
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
tf.logging.set_verbosity(tf.logging.INFO)


def load_data():
    train = []
    with tarfile.open('by.tar.xz') as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train.append(img)
    train = np.array(train)

    blue = []
    with tarfile.open('blue.tar.xz') as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_GRAYSCALE)
            blue.append(img)
    blue = np.array(blue)

    yellow = []
    with tarfile.open('yellow.tar.xz') as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_GRAYSCALE)
            yellow.append(img)
    yellow = np.array(yellow)

    train = train/train.max()
    blue = blue/blue.max()
    yellow = yellow/yellow.max()

    return train[100:], blue[100:], yellow[100:], train[:100], blue[:100], yellow[:100]

def build_model(X):
    conv1 = tf.layers.conv2d(X, 2,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             activation=tf.nn.relu,
                             name='conv1')
    return conv1

def train():
    model_name = "color_test"
    train, blue, yellow, test, bluetest, yellowtest = load_data()

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    y_blue = tf.placeholder(tf.float32, [None, 192, 256], name='y_blue')
    y_yellow = tf.placeholder(tf.float32, [None, 192, 256], name='y_yellow')

    op = build_model(X)

    loss = tf.losses.mean_squared_error(blue, tf.reshape(op[:,:,:,0], [None, op.shape[1], op.shape[2]])) + \
            tf.losses.mean_squared_error(yellow, tf.reshape(op[:,:,:,1], [None, op.shape[1], op.shape[2]]))
    train_op = tf.train.AdadeltaOptimizer(10**-2).minimize(loss)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for ep in range(100):
            train = train[np.random.permutation(train.shape[0])]
            blue = blue[np.random.permutation(blue.shape[0])]
            yellow = yellow[np.random.permutation(yellow.shape[0])]

            idx = 0
            batch_size = 100
            while idx < len(train.shape[0]):
                train_data = train[idx:idx+batch_size]
                blue_data = blue[idx:idx+batch_size]
                yellow_data = yellow[idx:idx+batch_size]

                train_loss, *_ = sess.run([loss, train_op],
                                          feed_dict={
                                              X: train_data,
                                              y_blue: blue_data,
                                              y_yellow: yellow_data
                                          })
                idx+=batch_size
                test_loss, *_ = sess.run([loss, train_op],
                                         feed_dict={
                                             X: test,
                                             y_blue: bluetest,
                                             y_yellow: yellowtest
                                         }) 
                logger.info(f'epoch {ep} batch {idx} training loss {train_loss} test loss {test_loss}')

        saver.save(sess, os.path.join(os.getcwd(), model_name), latest_filename=f'{model_name}.latest.ckpt')

def plot():
    model_name = "color_test"
    train, blue, yellow, test, bluetest, yellowtest = load_data()

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    y_blue = tf.placeholder(tf.float32, [None, 192, 256], name='y_blue')
    y_yellow = tf.placeholder(tf.float32, [None, 192, 256], name='y_yellow')

    op = build_model(X)

    loss = tf.losses.mean_squared_error(blue, tf.reshape(op[:,:,:,0], [None, op.shape[1], op.shape[2]])) + \
            tf.losses.mean_squared_error(yellow, tf.reshape(op[:,:,:,1], [None, op.shape[1], op.shape[2]]))
    train_op = tf.train.AdadeltaOptimizer(10**-2).minimize(loss)

    saver = tf.train.Saver()

    idx = np.random.randint(0, train.shape[0])

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))
        test_loss, conv_layer = sess.run([loss, op],
                                        feed_dict={
                                            X: test[idx:idx+1],
                                            y_blue: bluetest[idx:idx+1],
                                            y_yellow: yellowtest[idx:idx+1]
                                        })
    logger.info(f'test loss {test_loss}')
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(test[idx:idx + 1].reshape(test.shape[1], test.shape[2], 3))
    plt.subplot(1, 3, 2)
    layer1 = conv_layer[:, :, :, 0].reshape(conv_layer.shape[1], conv_layer.shape[2])
    plt.imshow(layer1, cmap='binary')
    plt.subplot(1, 3, 3)
    layer2 = conv_layer[:, :, :, 1].reshape(conv_layer.shape[1], conv_layer.shape[2])
    plt.imshow(layer2, cmap='binary')
    plt.show()


if __name__ == '__main__':
    train()
    plot()
