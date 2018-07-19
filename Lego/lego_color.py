import cv2
import logging
import os
import tarfile

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
    with tarfile.open('18.rb.300x150.txz') as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
            train.append(img)
    train = np.array(train)
    train_ = np.fliplr(train)
    train = np.vstack((train, train_))

    blue = []
    with tarfile.open('18.blue.label.300x150.txz') as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_GRAYSCALE)
            blue.append(img)
    blue = np.array(blue)
    blue_ = np.fliplr(blue)
    blue = np.vstack((blue, blue_))

    red = []
    with tarfile.open('18.red.label.300x150.txz') as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_GRAYSCALE)
            red.append(img)
    red = np.array(red)
    red_ = np.fliplr(red)
    red = np.vstack((red, red_))

    train = train/train.max()
    blue = blue/blue.max()
    red = red/red.max()

    seq = np.random.permutation(train.shape[0])
    train = train[seq]
    red = red[seq]
    blue = blue[seq]

    return train[6:], blue[6:], red[6:], train[:6], blue[:6], red[:6]


def build_model(X):
    conv0 = tf.layers.conv2d(X, 16, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, name='conv0')
    conv1 = tf.layers.conv2d(conv0, 2,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='SAME',
                             # activation=tf.nn.relu,
                             name='conv1')
    return conv1


def train():
    model_name = "color_test"
    train, blue, red, test, blue_test, red_test = load_data()

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 150, 300, 3], name='X')
    y_blue = tf.placeholder(tf.float32, [None, 150, 300], name='y_blue')
    y_red = tf.placeholder(tf.float32, [None, 150, 300], name='y_red')

    op = build_model(X)

    loss = tf.losses.mean_squared_error(y_blue, op[:, :, :, 0]) + tf.losses.mean_squared_error(y_red, op[:, :, :, 1])
    train_op = tf.train.AdadeltaOptimizer(10**-1).minimize(loss)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(1000):
            seq = np.random.permutation(train.shape[0])
            train = train[seq]
            blue = blue[seq]
            red = red[seq]

            idx = 0
            batch_size = 6

            while idx<train.shape[0]:
                train_loss, *_ = sess.run([loss, train_op],
                                          feed_dict={
                                              X: train[idx:idx+batch_size],
                                              y_blue: blue[idx:idx+batch_size],
                                              y_red: red[idx:idx+batch_size]
                                          })

                test_loss, *_ = sess.run([loss, train_op],
                                          feed_dict={
                                              X: test,
                                              y_blue: blue_test,
                                              y_red: red_test
                                          })

                logger.info(f'epoch {ep} training loss {train_loss} testing loss {test_loss}')

                idx+=batch_size

        saver.save(sess, os.path.join(os.getcwd(), model_name),
                   latest_filename=f'{model_name}.latest.ckpt')


def plot():
    model_name = "color_test"
    train, blue, red = load_data()

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 192, 256, 1], name='X')
    y_blue = tf.placeholder(tf.float32, [None, 192, 256], name='y_blue')
    y_red = tf.placeholder(tf.float32, [None, 192, 256], name='y_red')

    op = build_model(X)

    loss = tf.losses.mean_squared_error(y_blue, op[:, :, :, 0]) + tf.losses.mean_squared_error(y_red, op[:, :, :, 1])

    saver = tf.train.Saver()

    plt.figure()
    for i in range(6):
        idx = np.random.randint(0, test.shape[0])

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))
            test_loss, conv_layer = sess.run([loss, op],
                                            feed_dict={
                                                X: test[idx:idx+1],
                                                y_blue: bluetest[idx:idx+1],
                                                y_red: yellowtest[idx:idx+1]
                                            })
        logger.info(f'test loss {test_loss}')
        # plt.figure()
        plt.subplot(6, 3, 1+3*i)
        plt.imshow(test[idx:idx + 1].reshape(test.shape[1], test.shape[2]), cmap='gray')
        plt.subplot(6, 3, 2+3*i)
        layer1 = conv_layer[:, :, :, 0].reshape(conv_layer.shape[1], conv_layer.shape[2])
        plt.imshow(layer1, cmap='binary')
        plt.subplot(6, 3, 3+3*i)
        layer2 = conv_layer[:, :, :, 1].reshape(conv_layer.shape[1], conv_layer.shape[2])
        plt.imshow(layer2, cmap='binary')
    plt.show()


if __name__ == '__main__':
    train()
    # plot()
