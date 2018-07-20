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


    labels = np.loadtxt('18.rb.300x150.label.txt', delimiter=',', dtype=np.int8)
    labels = np.reshape(labels, (-1, 150, 300, 3))
    train = train/train.max()

    seq = np.random.permutation(train.shape[0])
    train = train[seq]
    labels = labels[seq]

    return train[1:], labels[1:], train[0], labels[0]


def build_model(X):
    conv0 = tf.layers.conv2d(X, 16, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, name='conv0')
    conv1 = tf.layers.conv2d(conv0, 3,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='SAME',
                             name='conv1')
    return conv1


def train():
    model_name = "color_test"
    train, label, *_ = load_data()

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 150, 300, 3], name='X')
    y_label = tf.placeholder(tf.float32, [None, 150, 300, 3], name='y_label')

    op = build_model(X)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=op))
    train_op = tf.train.AdadeltaOptimizer(10**-1).minimize(loss)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(2000):

            train_loss, *_ = sess.run([loss, train_op],
                                      feed_dict={
                                          X: train,
                                          y_label: label
                                      })


            logger.info(f'epoch {ep} training loss {train_loss}')


        saver.save(sess, os.path.join(os.getcwd(), model_name),
                   latest_filename=f'{model_name}.latest.ckpt')


def plot():
    model_name = "color_test"
    train, label, test, test_label = load_data()
    test = np.reshape(test, (-1, *test.shape))
    test_label = np.reshape(test_label, (-1, *test_label.shape))

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 150, 300, 3], name='X')
    y_label = tf.placeholder(tf.float32, [None, 150, 300, 3], name='y_label')

    op = build_model(X)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=op))

    saver = tf.train.Saver()

    plt.figure()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))
        test_loss, conv_layer, prediction = sess.run([loss, op, tf.argmax(op, axis=-1)],
                                        feed_dict={
                                            X: test,
                                            y_label: test_label
                                        })
    logger.info(f'test loss {test_loss}')

    plt.subplot(1, 2, 1)
    plt.imshow(test.reshape(test.shape[1], test.shape[2], 3))
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.reshape((150, 300)))

    plt.show()


if __name__ == '__main__':
    train()
    plot()
