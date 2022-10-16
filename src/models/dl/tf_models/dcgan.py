"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:02
# @FileName: dcgan.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.gan import GanModel
import tensorflow as tf
import numpy as np
import pickle


class DcganModel(GanModel):
    """Deep Convolution Generative Adversarial Network"""
    def __init__(self, args, task_type=None):
        GanModel.__init__(self, args, task_type)
        self.d_variables = None
        self.g_variables = None
        self.d_y = None
        self.g_y = None
        self.init_net()
        self.g_loss, self.d_loss, self.g_train, self.d_train = self.build_model()

    def init_net(self):
        # Network Inputs
        self.g_x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='fake_image')
        self.d_x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels], name='real_image')
        # Build Targets (real or fake images)
        self.d_y = tf.placeholder(tf.int32, shape=[None])
        self.g_y = tf.placeholder(tf.int32, shape=[None])
        # self.weights = {
        #     'gen_hidden1': tf.Variable(self.glorot_init([self.noise_dim, self.gen_hidden_dim])),
        #     'gen_out': tf.Variable(self.glorot_init([self.gen_hidden_dim, self.image_dim])),
        #     'disc_hidden1': tf.Variable(self.glorot_init([self.image_dim, self.disc_hidden_dim])),
        #     'disc_out': tf.Variable(self.glorot_init([self.disc_hidden_dim, 1]))
        # }
        # self.biases = {
        #     'gen_hidden1': tf.Variable(tf.zeros([self.gen_hidden_dim])),
        #     'gen_out': tf.Variable(tf.zeros([self.image_dim])),
        #     'disc_hidden1': tf.Variable(tf.zeros([self.disc_hidden_dim])),
        #     'disc_out': tf.Variable(tf.zeros([1]))
        # }

    def build_model(self):
        # Build Generator Network
        # g_x = tf.reshape(self.g_x, shape=[self.height, self.width, self.channels])
        g_sample = self.generator(self.g_x)

        # Build 2 Discriminator Networks (one from real image input, one from generated samples)
        d_real = self.discriminator(self.d_x)
        d_fake = self.discriminator(g_sample, reuse=True)
        d_concat = tf.concat([d_real, d_fake], axis=0)

        # Build the stacked generator/discriminator
        stacked_gan = self.discriminator(g_sample, reuse=True)

        # Build Loss
        d_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=d_concat, labels=self.d_y))
        g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_gan, labels=self.g_y))

        # Build Optimizers
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # g_variables_initner = tf.global_variables_initializer()
        # l_variables_initner = tf.local_variables_initializer()
        # tables_initner = tf.tables_initializer()
        # tf.Session().run(g_variables_initner)
        # tf.Session().run(l_variables_initner)
        # tf.Session().run(tables_initner)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        # self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator Network Variables
        # self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        # Create training operations
        g_train_op = g_optimizer.minimize(g_loss, var_list=self.g_variables, global_step=self.global_step)
        d_train_op = d_optimizer.minimize(d_loss, var_list=self.d_variables, global_step=self.global_step)
        self.g_sample = g_sample
        return g_loss, d_loss, g_train_op, d_train_op

    # Generator Network
    def generator(self, x, reuse=False):
        # Input: Noise, Output: Image
        # with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  # must in here
        return x

    # Discriminator Network
    def discriminator(self, x, reuse=False):
        # Input: Image, Output: Prediction Real/Fake Image
        # with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  # must in here
        return x

    def make_train_batch_for_g(self, batch_x):
        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the discriminator are real images,
        # the other half are fake images (coming from the generator).
        batch_x = np.reshape(batch_x, newshape=[-1, self.height, self.width, self.channels])
        z = np.random.uniform(-1., 1., size=[self.batch_size, self.input_size])
        batch_d_y = np.concatenate(
            [np.ones([self.batch_size]), np.zeros([self.batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_g_y = np.ones([self.batch_size])

        # Training
        feed_dict = {
            self.d_x: batch_x,
            self.g_x: z,
            self.d_y: batch_d_y,
            self.g_y: batch_g_y
        }
        return feed_dict

    def generate_image(self, sess, image_path):
        sample_images = tf.placeholder(tf.float32, [None, self.input_size])
        g_output = self.generator(sample_images)
        sample_noise = np.random.uniform(-1, 1, size=(25, self.input_size))
        sess.run(tf.global_variables_initializer())     # must have in dcgan test, gan don't need!
        samples = sess.run(g_output, feed_dict={sample_images: sample_noise})
        with open('{}/samples.pkl'.format(image_path), 'wb') as f:
            pickle.dump(samples, f)
