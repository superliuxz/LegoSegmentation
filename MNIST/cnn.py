import logging
from multiprocessing.dummy import Pool as ThreadPool
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
tf.logging.set_verbosity(tf.logging.WARN)


class CNN:
    def __init__(self):
        normalize_func = lambda x: x/x.max() # lambda x: (x-x.mean())/x.std()
        # crop a random 24x24 by 28x28 based on https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
        feat_extract_func = lambda x: image.PatchExtractor(patch_size=(24, 24), max_patches=1, random_state=1)\
                                           .transform(x).reshape(-1, 24, 24, 1)
        self.train_data, self.train_label, self.test_data \
            = self.load_data(normalize_func=normalize_func, feat_extract=feat_extract_func)
        
        self.img_size = self.train_data.shape[1]
        # depth of the conv1 filters. 36=6*6 easy plotting
        self.conv1_filter_depth = 36
        self.conv1_filter_depth_sqrt = np.int(np.sqrt(self.conv1_filter_depth))
        # depth of the conv2 filters. 64=**8 easy plotting
        self.conv2_filter_depth = 64
        self.conv2_filter_depth_sqrt = np.int(np.sqrt(self.conv2_filter_depth))
        # fully connected feature size. 576=24*24 easy plotting
        self.fc_feat_size = 576
        self.fc_feat_size_sqrt = np.int(np.sqrt(self.fc_feat_size))

        tf.reset_default_graph()

        self.result = pd.DataFrame()

        self.x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 1st conv layer
        self.w1 = tf.Variable(tf.truncated_normal([3, 3, 1, self.conv1_filter_depth], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.conv1_filter_depth]), name='b1')
        # (-1, self.img_size, self.img_size, self.filter_size)
        self.h1 = tf.nn.relu(
            tf.nn.conv2d(self.x, self.w1, strides=[1, 1, 1, 1], padding='SAME') + self.b1, name='h1')
        # 1st pooling layer
        # (-1, self.img_size/2, self.img_size/2, self.filter_size)
        self.pool1 = tf.nn.max_pool(self.h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # 2nd conv layer
        self.w2 = tf.Variable(tf.truncated_normal([3, 3, self.conv1_filter_depth, self.conv2_filter_depth], stddev=0.1), name='w2')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.conv2_filter_depth]), name='b2')
        # (-1, self.img_size/2, self.img_size/2, self.filter_size)
        self.h2 = tf.nn.relu(
            tf.nn.conv2d(self.pool1, self.w2, strides=[1, 1, 1, 1], padding='SAME') + self.b2, name='h2')
        # 2nd pooling layer
        # (-1, self.img_size/4,self.img_size/4, self.filter_size)
        self.pool2 = tf.nn.max_pool(self.h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # fc1
        # after twice maxpooling (downsampling), self.img_size -> self.img_size/4
        self.w3 = tf.Variable(tf.truncated_normal([self.img_size//4*self.img_size//4*self.conv2_filter_depth, self.fc_feat_size], stddev=0.1), name='w3')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[self.fc_feat_size]), name='b3')
        self.flat_pool2 = tf.reshape(self.pool2, [-1, self.img_size//4*self.img_size//4*self.conv2_filter_depth], name='flat_pool2')
        # (-1, self.fc_feat_size)
        self.h3 = tf.nn.relu(tf.matmul(self.flat_pool2, self.w3) + self.b3, name='h3')
        self.h3_drop = tf.nn.dropout(self.h3, self.keep_prob, name='h3_drop')

        # fc2, output
        self.w4 = tf.Variable(tf.truncated_normal([self.fc_feat_size, 10], stddev=0.1), name='w4')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[10]), name='b4')
        # (-1, 10)
        self.y_pred = tf.add(tf.matmul(self.h3_drop, self.w4), self.b4, name='y_pred')

        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_pred),
            name='cross_entropy_loss'
        )
        self.training_step = tf.train.AdamOptimizer(10 ** -3).minimize(self.cross_entropy_loss, name='training_step')
        self.correct_pred = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

        self.saver = tf.train.Saver()

    def train(self, batch_size: int):
        """
        ~5 mins on a gtx1070
        """
        logger.info(f'start training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for cv_idx, (train_data, train_label, valid_data, valid_label) in enumerate(self.split_train_valid()):
                batch_idx = 0
                while batch_idx < train_data.shape[0]:
                    x_batch = train_data[batch_idx:batch_idx+batch_size, :, :, :]
                    y_batch = train_label[batch_idx:batch_idx+batch_size, :]
                    # training_step is a oper not a tensor, therefore no return value
                    training_loss, *_ = sess.run([self.cross_entropy_loss, self.training_step], feed_dict={
                        self.x: x_batch,
                        self.y: y_batch,
                        self.keep_prob: 1/3
                    })
                    valid_accuracy = self.accuracy.eval(feed_dict={
                        self.x: valid_data,
                        self.y: valid_label,
                        self.keep_prob: 1.0
                    })
                    logger.info(f'cv {cv_idx} batch {batch_idx}, '
                                f'training loss: {training_loss}, '
                                f'validation accuracy: {valid_accuracy}')
                    self.result = self.result.append({'train_loss': training_loss, 'valid_acc': valid_accuracy},
                                                     ignore_index=True
                                                     )
                    batch_idx += batch_size
            self.saver.save(sess, os.path.join(os.getcwd(), 'my_model'))
        self.result.to_csv('result.csv', index=False)
        logger.info(f'done')

    def load_data(self, normalize_func: Callable=None, feat_extract: Callable=None) -> (np.array, np.array, np.array):
        train = pd.read_csv('train.csv.gz', header=0)
        # onehot encode the labels
        enc = OneHotEncoder(sparse=False)
        train_label = enc.fit_transform(train.pop('label').values.reshape(len(train), 1))
        # reshape into 42000 images, 28 by 28, 1 channel
        train_data = train.values.reshape(-1, 28, 28, 1)
        # reshape into 28000 images, 28 by 28, 1 channel
        test_data = pd.read_csv('test.csv.gz', header=0).values.reshape(-1, 28, 28, 1)

        if normalize_func is not None:
            logger.info(f'normalize data...')
            pool = ThreadPool(2)
            result = pool.map(normalize_func, [train_data, test_data])
            pool.close()
            pool.join()
            train_data, test_data = result
            logger.info(f'done')

        if feat_extract is not None:
            logger.info(f'cropping 24x24 outta 28x28...')
            pool = ThreadPool(2)
            result = pool.map(feat_extract, [train_data, test_data])
            pool.close()
            pool.join()
            train_data, test_data = result
            logger.info(f'done')

        logger.debug(f'training data: {train_data.shape}')
        logger.debug(f'train label: {train_label.shape}')
        logger.debug(f'test data: {test_data.shape}')

        return train_data, train_label, test_data

    def split_train_valid(self, n: int=10, shuffle: bool=True):
        """
        10-fold CV
        """
        kfold = KFold(n_splits=n, shuffle=shuffle)
        for train_idx, valid_idx in kfold.split(self.train_data, self.train_label):
            yield self.train_data[train_idx], self.train_label[train_idx], \
                  self.train_data[valid_idx], self.train_label[valid_idx]

    def plot_training_result(self):
        if len(self.result) == 0:
            self.result = pd.read_csv('result.csv', header=0)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0, len(self.result)), self.result['train_loss'])
        plt.title('training loss')
        plt.xlabel('training steps')

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, len(self.result)), self.result['valid_acc'])
        plt.title('validation accuracy')
        plt.xlabel('training steps')
        plt.show()

    def plot_confusion_matrix(self):
        """
        plot the confusion matrix for random half of the training set
        """
        (train_data, train_label, *_), _ = self.split_train_valid(n=2)
        logger.info('compute confusion matrix for random half of the trainin set ...')
        with self.restore_session() as sess:
            pred = self.y_pred.eval(session=sess, feed_dict={
                self.x: train_data,
                self.keep_prob: 1.0
            })
        logger.info('done')
        cnf_mat = confusion_matrix(np.argmax(pred, 1), np.argmax(train_label, 1)).astype(np.int)

        labels = [str(i) for i in range(10)]
        _, ax = plt.subplots(1)
        plt.imshow(cnf_mat, cmap='Wistia')
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(10):
            for j in range(10):
                ax.text(i, j, cnf_mat[j, i], va='center', ha='center')
        plt.title('confusion matrix')
        plt.ylabel('truth')
        plt.xlabel('prediction')
        plt.show()

    def plot_misclassification(self):
        """
        plot the misclassified image for random half of the training set
        """
        (train_data, train_label, *_), _ = self.split_train_valid(n=2)
        logger.info('compute prediction for random half of the trainin set ...')
        with self.restore_session() as sess:
            pred = self.y_pred.eval(session=sess, feed_dict={
                self.x: train_data,
                self.keep_prob: 1.0
            })
        logger.info('done')
        true_label = np.argmax(train_label, 1)
        pred_label = np.argmax(pred, 1)

        false_pred = [i for i in range(len(true_label)) if true_label[i] != pred_label[i]]

        plt.figure()
        # plot first 25
        for i in range(0, 5):
            for j in range(0, 5):
                curr = i*5+j
                if curr < len(false_pred):
                    ax = plt.subplot(5, 5, curr+1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.title(f'{true_label[false_pred[curr]]}/{pred_label[false_pred[curr]]}')
                    plt.imshow(train_data[false_pred[curr]].reshape(self.img_size, self.img_size), cmap='binary')
        plt.tight_layout()
        plt.show()

    def plot_activation(self):
        """
        plot the activation function for conv and pool
        """
        img_num = np.random.randint(0, self.train_data.shape[0])
        logger.info('recompute activations for different layers ...')
        with self.restore_session() as sess:
            h1 = self.h1.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num+1],
                self.keep_prob: 1.0
            })
            pool1 = self.pool1.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num+1],
                self.keep_prob: 1.0
            })
            h2 = self.h2.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num + 1],
                self.keep_prob: 1.0
            })
            pool2 = self.pool2.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num + 1],
                self.keep_prob: 1.0
            })
            h3 = self.h3.eval(session=sess, feed_dict={
                self.x: self.train_data[img_num: img_num + 1],
                self.keep_prob: 1.0
            })
        logger.info('done')
        plt.figure(figsize=(15, 10))
        # original image
        plt.subplot(2, 3, 1)
        plt.imshow(self.train_data[img_num].reshape(self.img_size, self.img_size), cmap='binary')
        # conv1
        plt.subplot(2, 3, 2)
        plt.title(f'conv1 {h1.shape}')
        h1 = np.reshape(h1, (-1, self.img_size, self.img_size, self.conv1_filter_depth_sqrt, self.conv1_filter_depth_sqrt))
        # TODO: I actually don't know what these magical numbers are
        h1 = np.transpose(h1, (0, 3, 1, 4, 2))
        h1 = np.reshape(h1, (-1, self.conv1_filter_depth_sqrt*self.img_size, self.conv1_filter_depth_sqrt*self.img_size))
        plt.imshow(h1[0], cmap='binary')
        # pool1
        plt.subplot(2, 3, 3)
        plt.title(f'pool1 {pool1.shape}')
        pool1 = np.reshape(pool1, (-1, self.img_size//2, self.img_size//2, self.conv1_filter_depth_sqrt, self.conv1_filter_depth_sqrt))
        pool1 = np.transpose(pool1, (0, 3, 1, 4, 2))
        pool1 = np.reshape(pool1, (-1, self.conv1_filter_depth_sqrt*self.img_size//2, self.conv1_filter_depth_sqrt*self.img_size//2))
        plt.imshow(pool1[0], cmap='binary')
        # conv2
        plt.subplot(2, 3, 4)
        plt.title(f'conv2 {h2.shape}')
        h2 = np.reshape(h2, (-1, self.img_size//2, self.img_size//2, self.conv2_filter_depth_sqrt, self.conv2_filter_depth_sqrt))
        h2 = np.transpose(h2, (0, 3, 1, 4, 2))
        h2 = np.reshape(h2, (-1, self.conv2_filter_depth_sqrt*self.img_size//2, self.conv2_filter_depth_sqrt*self.img_size//2))
        plt.imshow(h2[0], cmap='binary')
        # pool2
        plt.subplot(2, 3, 5)
        plt.title(f'pool2 {pool2.shape}')
        pool2 = np.reshape(pool2, (-1, self.img_size//4, self.img_size//4, self.conv2_filter_depth_sqrt, self.conv2_filter_depth_sqrt))
        pool2 = np.transpose(pool2, (0, 3, 1, 4, 2))
        pool2 = np.reshape(pool2, (-1, self.conv2_filter_depth_sqrt*self.img_size//4, self.conv2_filter_depth_sqrt*self.img_size//4))
        plt.imshow(pool2[0], cmap='binary')
        # fc1
        plt.subplot(2, 3, 6)
        plt.title(f'fc1 {h3.shape}')
        h3 = np.reshape(h3, (-1, self.fc_feat_size_sqrt, self.fc_feat_size_sqrt))
        plt.imshow(h3[0], cmap='binary')

        plt.show()

    def write_submission(self):
        logger.info('compute predicion on test dataset...')
        with self.restore_session() as sess:
            pred = self.y_pred.eval(session=sess, feed_dict={
                self.x: self.test_data,
                self.keep_prob: 1
            })
        logger.info('done')
        logger.info('start writing submission file...')
        pred = np.argmax(pred, 1)
        np.savetxt(
            'submission.csv',
            np.c_[range(1, len(pred)+1), pred],
            delimiter=',',
            header='ImageId,Label',
            comments='',
            fmt='%d'
        )
        logger.info('done')

    def restore_session(self) -> tf.Session:
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'my_model.meta'))
        session = tf.Session()
        self.reload_tensors(tf.get_default_graph())
        saver.restore(session, tf.train.latest_checkpoint('./'))
        return session

    def reload_tensors(self, graph: tf.Graph):
        self.x = graph.get_tensor_by_name('x:0')
        self.y = graph.get_tensor_by_name('y:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')

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
        self.flat_pool2 = graph.get_tensor_by_name('flat_pool2:0')
        self.h3 = graph.get_tensor_by_name('h3:0')

        self.w4 = graph.get_tensor_by_name('w4:0')
        self.b4 = graph.get_tensor_by_name('b4:0')
        self.y_pred = graph.get_tensor_by_name('y_pred:0')

        self.cross_entropy_loss = graph.get_tensor_by_name('cross_entropy_loss:0')
        self.training_step = graph.get_operation_by_name('training_step')
        self.correct_pred = graph.get_tensor_by_name('correct_pred:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')


if __name__ == '__main__':
    cnn = CNN()
    cnn.train(100)
    cnn.plot_training_result()
    cnn.plot_confusion_matrix()
    cnn.plot_misclassification()
    cnn.plot_activation()
    cnn.write_submission()
