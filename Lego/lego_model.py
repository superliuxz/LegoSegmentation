import cv2
import logging
import os
import tarfile
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import transform
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
tf.logging.set_verbosity(tf.logging.INFO)


class LEGO:
    def __init__(self, **kwargs):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        self.input = kwargs.get('input')
        self.label = kwargs.get('label')
        self.random_rotate = kwargs.get('random_rotate')

        self.img_h = kwargs.get('img_h')
        self.img_w = kwargs.get('img_w')

        self.conv1_filter_depth = kwargs.get('conv1_depth')

        self.conv2_filter_depth = kwargs.get('conv2_depth')

        self.fc_feat_size = kwargs.get('fc_feat')

        self.learning_rate = kwargs.get('lr')
        self.conv1_size = kwargs.get('conv1_size')
        self.conv2_size = kwargs.get('conv2_size')

        tf.reset_default_graph()

        self.result = pd.DataFrame()

        self.x = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, 3], name='x')
        self.y = tf.placeholder(tf.float32, [None, 512], name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # 1st conv layer
        self.w1 = tf.Variable(tf.truncated_normal([self.conv1_size, self.conv1_size, 3, self.conv1_filter_depth], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.conv1_filter_depth]), name='b1')
        # (-1, self.img_h, self.img_w, self.conv1_filter_depth)
        self.h1 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.nn.conv2d(self.x, self.w1, strides=[1, 1, 1, 1], padding='SAME') + self.b1,
                training=self.is_training), name='h1')
        self.print_h1 = tf.Print(self.h1, [self.h1], 'h1: ')

        # 1st pooling layer
        # (-1, self.img_h/2, self.img_w/2, self.conv1_filter_depth)
        self.pool1 = tf.nn.max_pool(self.h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # 2nd conv layer
        self.w2 = tf.Variable(tf.truncated_normal(
            [self.conv2_size, self.conv2_size, self.conv1_filter_depth, self.conv2_filter_depth], stddev=0.1), name='w2')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.conv2_filter_depth]), name='b2')
        # (-1, self.img_h/2, self.img_w/2, self.conv2_filter_depth)
        self.h2 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.nn.conv2d(self.pool1, self.w2, strides=[1, 1, 1, 1], padding='SAME') + self.b2,
                training=self.is_training), name='h2')
        self.print_h2 = tf.Print(self.h2, [self.h2], 'h2: ')

        # 2nd pooling layer
        # (-1, self.img_h/4,self.w/4, self.conv2_filter_depth)
        self.pool2 = tf.nn.max_pool(self.h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        final_conv_d = self.conv2_filter_depth
        final_pool = self.pool2

        # fc1
        # after twice pooling (downsampling), (h,w) -> (h/4,w/4)
        self.w3 = tf.Variable(tf.truncated_normal(
            [self.img_h//4*self.img_w//4*final_conv_d, self.fc_feat_size], stddev=0.1), name='w3')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[self.fc_feat_size]), name='b3')
        self.flat_pool = tf.reshape(
            final_pool, [-1, self.img_h//4*self.img_w//4*final_conv_d], name='flat_pool')
        # (-1, self.fc_feat_size)
        self.h3 = tf.nn.relu(tf.matmul(self.flat_pool, self.w3) + self.b3, name='h3')

        # # dense layer for first channel
        # self.w_fc1 = tf.Variable(tf.truncated_normal(
        #     [self.img_h // 4 * self.img_w // 4, self.fc_feat_size], stddev=0.1), name='w_fc1')
        # self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.fc_feat_size]), name='b_fc1')
        # self.flat_pool_fc1 = tf.reshape(
        #     final_pool[:, :, :, 0], [-1, self.img_h // 4 * self.img_w // 4], name='flat_pool_fc1')
        # self.h_fc1 = tf.nn.relu(tf.matmul(self.flat_pool_fc1, self.w_fc1) + self.b_fc1,
        #                         name='h_fc1')
        #
        # # dense layer for second channel
        # self.w_fc2 = tf.Variable(tf.truncated_normal(
        #     [self.img_h // 4 * self.img_w // 4, self.fc_feat_size], stddev=0.1), name='w_fc2')
        # self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.fc_feat_size]), name='b_fc2')
        # self.flat_pool_fc2 = tf.reshape(
        #     final_pool[:, :, :, 1], [-1, self.img_h // 4 * self.img_w // 4], name='flat_pool_fc2')
        # self.h_fc2 = tf.nn.relu(tf.matmul(self.flat_pool_fc2, self.w_fc2) + self.b_fc2,
        #                         name='h_fc2')
        #
        # self.h_concat = tf.reshape(tf.concat([self.h_fc1, self.h_fc2], axis=1), [-1, self.fc_feat_size*2],
        #                            name='h_concat')
        # self.print_h_concat = tf.Print(self.h_concat, [self.h_concat], 'self.h_concat: ')

        # # fc2, output
        # self.w4 = tf.Variable(tf.truncated_normal([self.fc_feat_size, 512], stddev=0.1), name='w4')
        # self.b4 = tf.Variable(tf.constant(0.1, shape=[512]), name='b4')
        # # (-1, 512)
        # self.y_pred = tf.add(tf.matmul(self.h3_drop, self.w4), self.b4, name='y_pred')

        self.loss = tf.losses.mean_squared_error(self.y, self.h3)

        self.training_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name='training_step')

        # r2 coeff. of correl.
        residuals = tf.reduce_sum(tf.square(tf.subtract(self.y, self.h3)))
        total = tf.reduce_sum(tf.square(tf.subtract(self.y, tf.reduce_mean(self.y))))
        self.accuracy = tf.subtract(1.0, tf.div(residuals, total), name='accuracy')

        # self.cross_entropy_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.h_concat),
        #     name='cross_entropy_loss'
        # )
        # self.training_step = tf.train.AdamOptimizer(self.learning_rate) \
        #     .minimize(self.cross_entropy_loss, name='training_step')
        # self.correct_pred = tf.equal(tf.argmax(self.h_concat, 1), tf.argmax(self.y, 1), name='correct_pred')
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

        self.saver = tf.train.Saver(save_relative_paths=True)

    def train(self, **kwargs):
        self._load_data_if_not(kwargs.get('normalize_func'))
        batch = kwargs.get('batches')
        batch_size = kwargs.get('batch_size')
        keep_prob = kwargs.get('keep_prob')
        epoch = kwargs.get('epoch')
        model_name = kwargs.get('model_name')

        logger.info(f'start training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ep in range(epoch):
                for cv, (X_train, y_train, X_valid, y_valid) in enumerate(self.gen_batches(n=batch)):
                    batch_idx = 0
                    while batch_idx < X_train.shape[0]:
                        train_loss, *_ = sess.run([self.loss, self.training_step], feed_dict={
                            self.x: X_train[batch_idx:batch_idx+batch_size, :, :, :],
                            self.y: y_train[batch_idx:batch_idx+batch_size, :],
                            self.is_training: False
                        })
                        train_accuracy = self.accuracy.eval(feed_dict={
                            self.x: X_valid,
                            self.y: y_valid,
                            self.is_training: False
                        })
                        test_loss = self.loss.eval(feed_dict={
                            self.x: self.test_data,
                            self.y: self.test_label,
                            self.is_training: False
                        })
                        logger.info(f'epoch {ep} cv {cv} batch {batch_idx}, '
                                    f'train_loss: {train_loss}, '
                                    f'train_acc: {train_accuracy}, '
                                    f'test_loss: {test_loss}')
                        self.result = self.result.append({'train_loss': train_loss, 'test_loss': test_loss},
                                                         ignore_index=True)
                        batch_idx += batch_size
                if train_loss < 10:
                    break
            self.saver.save(sess, os.path.join(os.getcwd(), model_name), latest_filename=f'{model_name}.latest.ckpt')
        self.result.to_csv(f'{model_name}.result.csv', index=False)
        logger.info(f'done')

    def gen_batches(self, n: int):
        kfold = KFold(n_splits=n, shuffle=True)
        for train_idx, valid_idx in kfold.split(self.train_data, self.train_label):
            yield self.train_data[train_idx], self.train_label[train_idx], \
                  self.train_data[valid_idx], self.train_label[valid_idx]

    def _load_data_if_not(self, normalize_func: Callable):
        if self.train_data is None and self.train_label is None:
            self.train_data, self.train_label, self.test_data, self.test_label \
                = self._load_data(self.input, self.label, random_rotate=self.random_rotate, normalize_func=normalize_func)

    def _load_data(self, dataset: str, label: str, random_rotate: int, normalize_func: Callable) -> (np.array, np.array, np.array, np.array):
        train = []
        with tarfile.open(dataset) as tar:
            for f in tar.getmembers():
                bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
                img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                train.append(img)
        train_data = np.array(train)
        train_label = np.genfromtxt(label, delimiter=',')
        # train_label = train_label.reshape(-1, 512, 2)

        if random_rotate > 0:
            logger.info(f'rotate img...')
            rot_data = []
            for _ in range(random_rotate):
                for im in train_data:
                    rim = transform.rotate(im, np.random.randint(0, 360), order=3, preserve_range=True)
                    rot_data.append(rim)
                train_label = np.vstack((train_label, train_label))
            rot_data = np.array(rot_data)
            train_data = np.vstack((train_data, rot_data))
            logger.info(f'done')

        logger.info(f'normalize data...')
        train_data = normalize_func(train_data)
        logger.info(f'done')

        # logger.info(f'split training/testing...')
        # idx = train_data.shape[0]//10
        # test_data = train_data[:idx, :, :, :]
        # test_label = train_label[:idx]
        # train_data = train_data[idx:, :, :, :]
        # train_label = train_label[idx:]
        # logger.info(f'done')
        test_data = []
        with tarfile.open('20.rb.256x192.tar.xz') as tar:
            for f in tar.getmembers():
                bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
                img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_data.append(img)
        test_data = np.array(test_data)
        test_data = normalize_func(test_data)
        test_label = np.genfromtxt('20.rb.256x192.label.txt', delimiter=',')

        logger.info(f'training data: {train_data.shape}')
        logger.info(f'training label: {train_label.shape}')
        logger.info(f'testing data: {test_data.shape}')
        logger.info(f'testing label: {test_label.shape}')

        return train_data, train_label, test_data, test_label

    def pred_three_random_img(self, filename: str, model_name: str, normalize_func: Callable, vectorize: tuple=None):
        data = []
        with tarfile.open(filename) as tar:
            for f in tar.getmembers():
                bimg = np.array(bytearray(tar.extractfile(f).read()), dtype=np.uint8)
                img = cv2.imdecode(bimg, flags=cv2.IMREAD_ANYCOLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append(img)
        data = np.array(data)
        data = normalize_func(data)
        plt.figure(figsize=(15, 8))

        for i in range(3):
            idx = np.random.randint(0, data.shape[0])
            img = data[idx: idx+1]
            oimg = img.reshape(self.img_h, self.img_w, 3)
            with self.restore_session(model_name) as sess:
                res = self.h3.eval(session=sess, feed_dict={
                    self.x: img,
                })
            if vectorize:
                res[np.where(res <= vectorize[0])] = 0
                res[np.where(np.logical_and(vectorize[0] < res, res <= vectorize[1]))] = 100
                res[np.where(res > vectorize[1])] = 200
            res = np.reshape(res, (16, 32))
            plt.subplot(2, 3, 1 + i)
            plt.imshow(oimg)
            plt.subplot(2, 3, 4 + i)
            plt.imshow(res, cmap='binary')
        plt.show()

    def pred_test_img(self, images: List[str], model_name: str, normalize_func: Callable):
        """
        images is a list of images file names, i.e. ['1.jpg', '2.jpg'..]
        """
        plt.figure()
        for i, im in enumerate(images):
            img = cv2.imread(im, flags=cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            oimg = cv2.resize(src=img, dsize=(self.img_w, self.img_h), interpolation=cv2.INTER_LANCZOS4)
            img = oimg.reshape(-1, self.img_h, self.img_w, 3)
            img = normalize_func(img)
            with self.restore_session(model_name) as sess:
                res = self.h3.eval(session=sess, feed_dict={
                    self.x: img,
                })
            res = np.reshape(res, (16, 32))

            plt.subplot(2, len(images), 1+i)
            plt.imshow(oimg)
            plt.subplot(2, len(images), 1+len(images)+i)
            plt.imshow(res, cmap='binary')
        plt.show()

    def plot_training_result(self, model_name: str):
        if len(self.result) == 0:
            self.result = pd.read_csv(f'{model_name}.result.csv', header=0)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0, len(self.result)), self.result['train_loss'])
        plt.title('train_loss')
        plt.xlabel('training steps')

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, len(self.result)), self.result['test_loss'])
        plt.title('test_loss')
        plt.xlabel('training steps')
        plt.show()

    def plot_activation(self, model_name: str, normalize_func: Callable):
        """
        plot the activation function for conv and pool
        """
        def closest_factors(n: int) -> (int, int):
            x = int(np.sqrt(n))
            while n % x != 0:
                x -= 1
            return (x, n//x) if x > n else (n//x, x)

        self._load_data_if_not(normalize_func)

        img_num = np.random.randint(0, self.test_data.shape[0]-1)
        logger.info('recompute activations for different layers ...')
        with self.restore_session(model_name) as sess:
            h1 = self.h1.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num+1],
            })
            pool1 = self.pool1.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num+1],
            })
            h2 = self.h2.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num + 1],
            })
            pool2 = self.pool2.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num + 1],
            })
            h3 = self.h3.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num + 1],
            })
        logger.info('done')

        plt.figure(figsize=(15, 10))
        # original image
        plt.subplot(2, 3, 1)
        plt.title(f'original image (normalized)')
        plt.imshow(self.train_data[img_num])

        c1_h, c1_w = closest_factors(self.conv1_filter_depth)
        c2_h, c2_w = closest_factors(self.conv2_filter_depth)
        fc_h, fc_w = closest_factors(self.fc_feat_size)

        # conv1
        plt.subplot(2, 3, 2)
        plt.title(f'conv1 {h1.shape}')
        h1 = np.reshape(h1, (-1, self.img_h, self.img_w, c1_h, c1_w))
        h1 = np.transpose(h1, (0, 3, 1, 4, 2))
        h1 = np.reshape(h1, (-1, c1_h * self.img_h, c1_w * self.img_w))
        plt.imshow(h1[0])

        # pool1
        plt.subplot(2, 3, 3)
        plt.title(f'pool1 {pool1.shape}')
        pool1 = np.reshape(pool1, (-1, self.img_h // 2, self.img_w // 2, c1_h, c1_w))
        pool1 = np.transpose(pool1, (0, 3, 1, 4, 2))
        pool1 = np.reshape(pool1, (-1, c1_h * self.img_h // 2, c1_w * self.img_w // 2))
        plt.imshow(pool1[0])

        # conv2
        plt.subplot(2, 3, 4)
        plt.title(f'conv2 {h2.shape}')
        h2 = np.reshape(h2, (-1, self.img_h // 2, self.img_w // 2, c2_h, c2_w))
        h2 = np.transpose(h2, (0, 3, 1, 4, 2))
        h2 = np.reshape(h2, (-1, c2_h * self.img_h // 2, c2_w * self.img_w // 2))
        plt.imshow(h2[0], cmap='binary')

        # pool2
        plt.subplot(2, 3, 5)
        plt.title(f'pool2 {pool2.shape}')
        pool2 = np.reshape(pool2, (-1, self.img_h // 4, self.img_w // 4, c2_h, c2_w))
        pool2 = np.transpose(pool2, (0, 3, 1, 4, 2))
        pool2 = np.reshape(pool2, (-1, c2_h * self.img_h // 4, c2_w * self.img_w // 4))
        plt.imshow(pool2[0], cmap='binary')

        # fc1
        plt.subplot(2, 3, 6)
        plt.title(f'fc1 {h3.shape}')
        h3 = np.reshape(h3, (-1, fc_w, fc_h))
        plt.imshow(h3[0], cmap='binary')

        plt.show()

    def plot_vectorized_prediction_mse_on_training(self, model_name: str, normalize_func: Callable, xrange: Tuple[int, int], yrange: Tuple[int, int]):
        """
        apply vectorization onto the prediction of training set, then recompute MSE, also plot the contour of new MSE
        for lo in xrange:
          for hi in yrange:
            x<=lower -> x=0
            lower<x<=upper -> x=100
            upper<x -> x=200
        """
        self._load_data_if_not(normalize_func)

        with self.restore_session(model_name) as sess:
            batch_idx = 0
            res = []
            while batch_idx < self.train_data.shape[0]:
                res.append(self.h3.eval(session=sess, feed_dict={
                    self.x: self.train_data[batch_idx:batch_idx+100],
                }))
                batch_idx += 100
        res = np.array(res).reshape(-1, 512)
        self._plot_new_mse(res, self.train_label, xrange, yrange)

    def plot_vectorized_prediction_mse_on_testing(self, filename: str, labelname: str, model_name: str, normalize_func: Callable, xrange: Tuple[int, int], yrange: Tuple[int, int]):
        """
        apply vectorization onto the prediction of testing set, then recompute MSE, also plot the contour of new MSE
        for lo in xrange:
          for hi in yrange:
            x<=lower -> x=0
            lower<x<=upper -> x=100
            upper<x -> x=200
        """
        train, label, *_ = self._load_data(filename, labelname, 0, normalize_func)

        with self.restore_session(model_name) as sess:
            res = self.h3.eval(session=sess, feed_dict={
                self.x: train,
            })
        self._plot_new_mse(res, label, xrange, yrange)

    def _plot_new_mse(self, prediction: np.array, ground_truth: np.array, xrange: Tuple[int, int], yrange: Tuple[int, int]):
        MSEs = []
        x, y = np.arange(xrange[0], xrange[1], step=1), np.arange(yrange[0], yrange[1], step=1)
        for lo in x:
            for hi in y:
                rep = np.copy(prediction)
                rep[np.where(rep <= lo)] = 0
                rep[np.where(np.logical_and(lo < rep, rep <= hi))] = 100
                rep[np.where(rep > hi)] = 200
                MSEs.append(mean_squared_error(ground_truth, rep))
        MSEs = np.array(MSEs)
        MSEs = MSEs.reshape(yrange[1]-yrange[0], xrange[1]-xrange[0])  # y, x
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('x<=lower -> x=0; lower<x<=upper -> x=100; upper<x -> x=200')
        ax.set_xlabel('lower threshold')
        ax.set_ylabel('higher threshold')
        cnt = plt.contour(X, Y, MSEs)
        plt.clabel(cnt, inline=1, fontsize=10)
        plt.show()

    def restore_session(self, model_name: str) -> tf.Session:
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), f'{model_name}.meta'))
        session = tf.Session()
        self.reload_tensors(tf.get_default_graph())
        saver.restore(session, tf.train.latest_checkpoint(os.getcwd(), latest_filename=f'{model_name}.latest.ckpt'))
        return session

    def reload_tensors(self, graph: tf.Graph):
        self.x = graph.get_tensor_by_name('x:0')
        self.y = graph.get_tensor_by_name('y:0')

        self.w1 = graph.get_tensor_by_name('w1:0')
        self.b1 = graph.get_tensor_by_name('b1:0')
        self.h1 = graph.get_tensor_by_name('h1:0')
        self.pool1 = graph.get_tensor_by_name('pool1:0')

        self.w2 = graph.get_tensor_by_name('w2:0')
        self.b2 = graph.get_tensor_by_name('b2:0')
        self.h2 = graph.get_tensor_by_name('h2:0')
        self.pool2 = graph.get_tensor_by_name('pool2:0')

        self.w3 = graph.get_tensor_by_name('w3:0')
        self.b3 = graph.get_tensor_by_name('b3:0')
        self.flat_pool = graph.get_tensor_by_name('flat_pool:0')
        self.h3 = graph.get_tensor_by_name('h3:0')

        # self.w4 = graph.get_tensor_by_name('w4:0')
        # self.b4 = graph.get_tensor_by_name('b4:0')
        # self.y_pred = graph.get_tensor_by_name('y_pred:0')

        # self.w_fc1 = graph.get_tensor_by_name('w_fc1:0')
        # self.b_fc1 = graph.get_tensor_by_name('b_fc1:0')
        # self.flat_pool_fc1 = graph.get_tensor_by_name('flat_pool_fc1:0')
        # self.h_fc1 = graph.get_tensor_by_name('h_fc1:0')
        #
        # self.w_fc2 = graph.get_tensor_by_name('w_fc2:0')
        # self.b_fc2 = graph.get_tensor_by_name('b_fc2:0')
        # self.flat_pool_fc1 = graph.get_tensor_by_name('flat_pool_fc2:0')
        # self.h_fc2 = graph.get_tensor_by_name('h_fc2:0')
        #
        # self.h_concat = graph.get_tensor_by_name('h_concat:0')

        # self.cross_entropy_loss = graph.get_tensor_by_name('cross_entropy_loss:0')
        # self.training_step = graph.get_operation_by_name('training_step')
        # self.correct_pred = graph.get_tensor_by_name('correct_pred:0')
        # self.accuracy = graph.get_tensor_by_name('accuracy:0')

        self.training_step = graph.get_operation_by_name('training_step')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')


if __name__ == '__main__':
    kw = {
        'img_w': 256,
        'img_h': 192,
        'conv1_depth': 16,
        'conv1_size': 3,
        'conv2_depth': 32,
        'conv2_size': 3,
        'fc_feat': 512,
        'lr': 10**-3,
        'input': '5k.rb.256x192.tar.xz',
        'label': '5k.rb.256x192.label.txt',
        'random_rotate': 0,
    }

    lego = LEGO(**kw)

    train_param = {
        'keep_prob': 1,
        'batches': 10,
        'batch_size': 50,
        'epoch': 6,
        'model_name': '5k_rb',
        'normalize_func': lambda x: x / x.max()
    }

    lego.train(**train_param)

    lego.plot_training_result(model_name=train_param.get('model_name'))

    lego.plot_activation(model_name=train_param.get('model_name'),
                         normalize_func=train_param.get('normalize_func'))

    # lego.pred_test_img(images=['r.test.jpg', 'r.test2.jpg', 'r.test3.jpg'],
    #                    model_name=train_param.get('model_name'),
    #                    normalize_func=train_param.get('normalize_func'))

    # lego.pred_three_random_img(filename='20.rb.256x192.tar.xz',
    #                            model_name=train_param.get('model_name'),
    #                            normalize_func=train_param.get('normalize_func'),
    #                            vectorize=None)

    # lego.plot_vectorized_prediction_mse_on_training(model_name=train_param.get('model_name'),
    #                                                 normalize_func=train_param.get('normalize_func'),
    #                                                 xrange=(25, 75),
    #                                                 yrange=(125, 175))

    # lego.plot_vectorized_prediction_mse_on_testing(filename='100.by.256x192.tar.xz',
    #                                                labelname='100.by.256x192.label.txt',
    #                                                model_name=train_param.get('model_name'),
    #                                                normalize_func=train_param.get('normalize_func'),
    #                                                xrange=(25, 75),
    #                                                yrange=(125, 175))
