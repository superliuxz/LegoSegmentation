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
                                       strides=(5, 5),
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
    # simple 1x1 network
    conv1_ = tf.layers.conv2d(conv3, 3,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='SAME',
                             name='1x1_conv1')
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
                                         strides=(5, 5),
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         name='deconv3')
    return conv3, conv1_, deconv3


def load_data(dataset: str, label: str, normalize_func: Callable) -> (np.array, np.array):
    train = []
    with tarfile.open(dataset) as tar:
        for f in tar.getmembers():
            bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
            img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train.append(img)
    train_data = np.array(train)

    train_label = np.genfromtxt(label, delimiter=',')
    train_label = np.reshape(train_label, (-1, 15, 30, 3))

    logger.info(f'normalize data...')
    train_data = normalize_func(train_data)
    logger.info(f'done')

    logger.info(f'training data: {train_data.shape}')
    logger.info(f'training label: {train_label.shape}')

    seq = np.random.permutation(train_data.shape[0])
    train_data = train_data[seq]
    train_label = train_label[seq]

    return train_data[1:], train_label[1:], train_data[0], train_label[0]


def train():
    tf.reset_default_graph()
    model_name = 'lego_fcn'

    train_data, train_label, test_data, test_label = load_data(dataset='18.rb.300x150.txz',
                                                               label='18.rb.300x150.label.txt',
                                                               normalize_func=lambda x: x/x.max())

    X = tf.placeholder(tf.float32, [None, 150, 300, 3], name='X')
    y = tf.placeholder(tf.float32, [None, 15, 30, 3], name='y_label')

    middle, encode_op, decode_op = build_model(X)

    l2_loss = tf.losses.mean_squared_error(X, decode_op)
    crx_entr_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=encode_op)
    )
    loss = l2_loss+crx_entr_loss
    train_op = tf.train.AdadeltaOptimizer(10**-2).minimize(loss)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(5000):


            train_l2, train_crx, *_ = sess.run([l2_loss, crx_entr_loss, train_op],
                                      feed_dict={
                                          X: train_data,
                                          y: train_label
                                      })

            logger.info(f'epoch {i} training l2 loss {train_l2} training crx entr loss {train_crx}')

        saver.save(sess, os.path.join(os.getcwd(), model_name), latest_filename=f'{model_name}.latest.ckpt')

def plot():
    tf.reset_default_graph()
    model_name = 'lego_fcn'

    train_data, train_label, test_data, test_label = load_data(dataset='18.rb.300x150.txz',
                                                               label='18.rb.300x150.label.txt',
                                                               normalize_func=lambda x: x / x.max())
    test_data = np.reshape(test_data, (-1, *test_data.shape))
    test_label = np.reshape(test_label, (-1, *test_label.shape))

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 150, 300, 3], name='X')
    y_label = tf.placeholder(tf.float32, [None, 15, 30, 3], name='y_label')

    middle, encode_op, decode_op = build_model(X)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=encode_op))

    saver = tf.train.Saver()

    plt.figure()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))
        test_loss, conv_layer, prediction = sess.run([loss, middle, tf.argmax(encode_op, axis=-1)],
                                        feed_dict={
                                            X: test_data,
                                            y_label: test_label
                                        })
    logger.info(f'test loss {test_loss}')

    plt.subplot(1, 3, 1)
    plt.imshow(test_data.reshape(test_data.shape[1], test_data.shape[2], 3))
    plt.subplot(1, 3, 2)
    plt.imshow(conv_layer.reshape((15, 30)), cmap='binary')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction.reshape((15, 30)))

    plt.show()

def test_encode():
    model_name = 'lego_fcn'
    tf.reset_default_graph()

    # *_, test_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    # train_data = load_data(dataset='20.rb.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    train_data = load_data(dataset='100.by.256x192.tar.xz', normalize_func=lambda x: x / x.max())
    # train_data = np.vstack((train_data, train_data2))

    X = tf.placeholder(tf.float32, [None, 192, 256, 3], name='X')
    y = tf.placeholder(tf.float32, [None, 1024], name='y')
    encode_op, decode_op = build_model(X)
    saver = tf.train.Saver()

    idx = np.random.randint(0, train_data.shape[0])

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))

        middle_layer, *_ = sess.run([encode_op],
                                    feed_dict={
                                        X: train_data[idx:idx+1]
                                    })
    p1 = np.reshape(middle_layer[0][:512], [16, 32])
    p1 = 1 / (1 + np.exp(-p1))
    p2 = np.reshape(middle_layer[0][512:], [16, 32])
    p2 = 1 / (1 + np.exp(-p2))
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
    y = tf.placeholder(tf.float32, [None, 1024], name='y')
    encode_op, decode_op = build_model(X)
    loss = tf.losses.mean_squared_error(X, decode_op) + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=encode_op)
    )
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
        cv2.imwrite(f'{i:02d}.jpg', 255 - layer.reshape(layer.shape[0], layer.shape[1]) * 255)


if __name__ == '__main__':
    train()
    plot()
    # test_encode()
    # test_decode()
    # save_midddle_to_file()
