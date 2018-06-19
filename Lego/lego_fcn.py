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

# np.random.seed(42)
# tf.set_random_seed(42)


def build_model(X):
    # encoder
    conv1 = tf.layers.conv2d(X, 16,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             activation=tf.nn.relu,
                             name='conv1')
    maxpool1 = tf.layers.max_pooling2d(conv1,
                                       pool_size=(2, 2),
                                       strides=(2, 2),
                                       padding='SAME',
                                       name='pool1')
    conv2 = tf.layers.conv2d(maxpool1, 32,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             activation=tf.nn.relu,
                             name='conv2')
    maxpool2 = tf.layers.max_pooling2d(conv2,
                                       pool_size=(2, 2),
                                       strides=(2, 2),
                                       padding='SAME',
                                       name='pool2')
    conv3 = tf.layers.conv2d(maxpool2, 1,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             activation=tf.nn.relu,
                             name='conv3')
    # decoder
    deconv1 = tf.layers.conv2d_transpose(conv3, 32,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         name='deconv1')
    deconv2 = tf.layers.conv2d_transpose(deconv1, 16,
                                         kernel_size=(3, 3),
                                         strides=(2, 2),
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         name='deconv2')
    deconv3 = tf.layers.conv2d_transpose(deconv2, 3,
                                         kernel_size=(3, 3),
                                         strides=(2, 2),
                                         padding='SAME',
                                         name='deconv3')
    return conv3, deconv3


def load_data(dataset: str, normalize_func: Callable) -> (np.array, np.array):
    train = []
    with tarfile.open(dataset) as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train.append(img)
    train_data = np.array(train)

    logger.info(f'normalize data...')
    train_data = normalize_func(train_data)
    logger.info(f'done')

    #idx = train_data.shape[0] // 5
    #test_data = train_data[:idx]
    #train_data = train_data[idx:]

    logger.info(f'training data: {train_data.shape}')

    # return train_data, test_data
    return train_data


def train():
    tf.reset_default_graph()
    model_name = 'lego_fcn'

    # train_data, test_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x/x.max())
    # train_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    train_data = load_data(dataset='100.by.256x192.tar.xz', normalize_func=lambda x: x/x.max())
    # train_data = np.vstack((train_data, train_data2))

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    encode_op, decode_op = build_model(X)

    loss = tf.losses.mean_squared_error(X, decode_op)
    train_op = tf.train.AdadeltaOptimizer(10**-2).minimize(loss)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(5000):
            train_data = train_data[np.random.permutation(train_data.shape[0])]

            batch_idx = 0
            batch_size = 20
            while batch_idx < train_data.shape[0]:
                X_ = train_data[batch_idx:batch_idx+batch_size]

                train_loss, *_ = sess.run([loss, train_op],
                                          feed_dict={
                                              X: X_
                                          })
                batch_idx += batch_size
                # test_loss, *_ = sess.run([loss, train_op],
                #                          feed_dict={
                #                             X: test_data
                #                          })
            #if i % 100 == 0:
            logger.info(f'epoch {i} training loss {train_loss}')


        saver.save(sess, os.path.join(os.getcwd(), model_name), latest_filename=f'{model_name}.latest.ckpt')


def test_encode():
    model_name = 'lego_fcn'
    tf.reset_default_graph()

    # *_, test_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    # train_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    train_data = load_data(dataset='100.by.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    # train_data = np.vstack((train_data, train_data2))

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    encode_op, decode_op = build_model(X)
    saver = tf.train.Saver()

    idx = np.random.randint(0, train_data.shape[0])

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))

        middle_layer, *_ = sess.run([encode_op],
                                    feed_dict={
                                        X: train_data[idx:idx+1]
                                    })
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(train_data[idx:idx + 1].reshape(train_data.shape[1], train_data.shape[2], 3))
    plt.subplot(1, 2, 2)
    plt.imshow(np.reshape(middle_layer[:, :, :, 0], (middle_layer.shape[1], middle_layer.shape[2])), cmap='binary')
    plt.show()


def test_decode():
    tf.reset_default_graph()
    model_name = 'lego_fcn'

    # *_, test_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    # train_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    train_data = load_data(dataset='100.by.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    # train_data = np.vstack((train_data, train_data2))

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    encode_op, decode_op = build_model(X)
    loss = tf.losses.mean_squared_error(X, decode_op)
    saver = tf.train.Saver()

    idx = np.random.randint(0, train_data.shape[0])

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))

        final_layer, loss, *_ = sess.run([decode_op, loss],
                                         feed_dict={
                                             X: train_data[idx:idx+1]
                                         })
    print(loss)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(train_data[idx:idx + 1].reshape(train_data.shape[1], train_data.shape[2], 3))
    plt.subplot(1, 2, 2)
    final_layer = final_layer.reshape(final_layer.shape[1], final_layer.shape[2], 3)
    final_layer = (final_layer * 255).astype(np.uint8)
    img = Image.fromarray(final_layer, 'RGB')
    plt.imshow(img)
    plt.show()


def save_midddle_to_file():
    tf.reset_default_graph()
    model_name = 'lego_fcn'

    train_data = load_data(dataset='100.by.256x192.tar.xz', normalize_func=lambda x: x / x.max())

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    encode_op, decode_op = build_model(X)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))

        middle_layer, *_ = sess.run([encode_op],
                                    feed_dict={
                                        X: train_data
                                    })
    for i, layer in enumerate(middle_layer):
        print(layer.shape)
        img = Image.fromarray(layer.reshape(layer.shape[0], layer.shape[1]), 'L')
        img.save(f'{i}.jpg')

if __name__ == '__main__':
    train()
    # test_encode()
    # test_decode()
    save_midddle_to_file()
