"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:01
# @FileName: gan.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.model import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


class GanModel(Model):
    """Generative Adversarial Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
        self.height = 28
        self.width = 28
        self.channels = 1
        self.gen_hidden_dim = 256
        self.disc_hidden_dim = 256
        self.g_x = None
        self.d_x = None
        self.g_sample = None
        self.init_net()
        self.g_loss, self.d_loss, self.g_train, self.d_train = self.build_model()

    def init_net(self):
        # Network Inputs
        self.d_x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='real_image')
        self.g_x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='fake_image')
        self.weights = {
            'gen_hidden1': tf.Variable(self.glorot_init([self.input_size, self.gen_hidden_dim])),
            'gen_out': tf.Variable(self.glorot_init([self.gen_hidden_dim, self.input_size])),
            'disc_hidden1': tf.Variable(self.glorot_init([self.input_size, self.disc_hidden_dim])),
            'disc_out': tf.Variable(self.glorot_init([self.disc_hidden_dim, 1]))
        }
        self.biases = {
            'gen_hidden1': tf.Variable(tf.zeros([self.gen_hidden_dim])),
            'gen_out': tf.Variable(tf.zeros([self.input_size])),
            'disc_hidden1': tf.Variable(tf.zeros([self.disc_hidden_dim])),
            'disc_out': tf.Variable(tf.zeros([1]))
        }

    def build_model(self):
        # Build Generator Network
        g_sample = self.generator(self.g_x)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        d_real = self.discriminator(self.d_x)
        d_fake = self.discriminator(g_sample)

        # Build Loss
        g_loss = -tf.reduce_mean(tf.log(d_fake), keep_dims=True)
        d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake), keep_dims=True)

        # Build Optimizers
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        g_vars = [
            self.weights['gen_hidden1'], self.weights['gen_out'],
            self.biases['gen_hidden1'], self.biases['gen_out']
        ]
        # Discriminator Network Variables
        d_vars = [
            self.weights['disc_hidden1'], self.weights['disc_out'],
            self.biases['disc_hidden1'], self.biases['disc_out']
        ]

        # Create training operations
        g_train = g_optimizer.minimize(g_loss, var_list=g_vars, global_step=self.global_step)
        d_train = d_optimizer.minimize(d_loss, var_list=d_vars, global_step=self.global_step)
        self.g_sample = g_sample
        return g_loss, d_loss, g_train, d_train

    def fit_unsupervised(self, sess, feed_dict):
        """
        learn model, 训练模型
        :param sess:
        :param feed_dict:
        :return:
        """
        g_loss, d_loss, _, _, step = sess.run(
            [self.g_loss, self.d_loss, self.g_train, self.d_train, self.global_step],
            feed_dict=feed_dict
        )
        return g_loss, d_loss, step

    # Generator
    def generator(self, x):
        hidden_layer = tf.matmul(x, self.weights['gen_hidden1'])
        hidden_layer = tf.add(hidden_layer, self.biases['gen_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.matmul(hidden_layer, self.weights['gen_out'])
        out_layer = tf.add(out_layer, self.biases['gen_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    # Discriminator
    def discriminator(self, x):
        hidden_layer = tf.matmul(x, self.weights['disc_hidden1'])
        hidden_layer = tf.add(hidden_layer, self.biases['disc_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.matmul(hidden_layer, self.weights['disc_out'])
        out_layer = tf.add(out_layer, self.biases['disc_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    def make_train_batch_for_g(self, batch_x):
        # Generate noise to feed to the generator
        # z = tf.random.uniform([self.batch_size, self.noise_dim], -1., 1.)
        z = np.random.uniform(-1., 1., size=[self.batch_size, self.input_size])
        feed_dict = {
            self.d_x: batch_x,
            self.g_x: z
        }
        return feed_dict

    def generate_image(self, sess, image_path):
        sample_images = tf.placeholder(tf.float32, [None, self.input_size])
        g_output = self.generator(sample_images)
        sample_noise = np.random.uniform(-1, 1, size=(25, self.input_size))
        samples = sess.run(g_output, feed_dict={sample_images: sample_noise})
        with open('{}/samples.pkl'.format(image_path), 'wb') as f:
            pickle.dump(samples, f)

    def show(self, image_path):
        with open('{}/samples.pkl'.format(image_path), 'rb') as f:
            samples = pickle.load(f)
        fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
        for ax, image in zip(axes.flatten(), samples):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(image.reshape((self.height, self.width)), cmap='Greys_r')
        plt.show()

    @staticmethod
    def glorot_init(shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
