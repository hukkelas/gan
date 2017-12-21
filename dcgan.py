import tensorflow as tf
from operations import fully_connected, deconv2d, conv2d, lrelu
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
from utils import save_image
import time
class DCGAN:

  def __init__(self, X_train):
    self.images = input_data.read_data_sets('MNIST_data/').train
    #assert images.shape[1] == self.images.shape[2], "Expected a NxN image. Got shape: {}".format(self.images.shape[1:3])
    self.imsize, self.c_dim = [28,1]
    self.z_dim = 100
    self.learning_rate = 0.002
    self.adam_beta = 0.5
    self.sample_num = 10
    self.batch_size = 64
    self.build_model(self.batch_size)

  def generator(self, z, batch_size, reuse_variables=False):
    start_dim = 4
    with tf.variable_scope("generator") as scope:
      if reuse_variables:
        scope.reuse_variables()
      
      # Fully connected to reshape Z 
      h0 = fully_connected(z, 1024*start_dim**2, tf.nn.relu, True, "g_fc_0")
      h0 = tf.reshape(h0, [batch_size, start_dim, start_dim, 1024])

      # Deconv 1
      h1 = deconv2d(h0, [batch_size, start_dim*2, start_dim*2, 512],
        scope="g_deconv2d_h1",
        batch_norm=True,
        activation_fn=tf.nn.relu)

      # Deconv layer 2
      h2 = deconv2d(h1,[batch_size, start_dim*4, start_dim*4, 256],
        scope="g_deconv2d_h2",
        batch_norm=True,
        activation_fn=tf.nn.relu)
      
      # Deconv layer 3
      h3 = deconv2d(h2, [batch_size, start_dim*8, start_dim*8, 128],
        scope="g_deconv2d_h3",
        batch_norm=True,
        activation_fn=tf.nn.relu)
      
      # Deconv layer 4
      #h4 = deconv2d(h3, [batch_size, start_dim*16, start_dim*16, 64],
      #  scope="g_deconv2d_h4",
      #  batch_norm=True,
      #  activation_fn=tf.nn.relu)
      
      h4 = tf.reshape(h3, [batch_size, -1])
      h5 = fully_connected(h4, self.imsize*self.imsize*self.c_dim,activation_fn=tf.nn.tanh, scope="g_fc_h5")
      h5 = tf.reshape(h5,[batch_size, self.imsize, self.imsize, self.c_dim])
      return h5
  
  def discriminator(self, x, batch_size, reuse_variables=False):
    d = 64
    with tf.variable_scope('discriminator') as scope:
      if reuse_variables:
        scope.reuse_variables()
      # Convolve into 4*4*1024
      h0 = tf.reshape(x, [batch_size, self.imsize, self.imsize, self.c_dim])
      h0 = conv2d(h0, d, scope="d_conv2d_h0", activation_fn=lrelu, batch_norm=False)
      h1 = conv2d(h0, d*2, scope="d_conv2d_h1", activation_fn=lrelu, batch_norm=True)
      h2 = conv2d(h1, d*4, scope="d_conv2d_h2", activation_fn=lrelu, batch_norm=True)
      h3 = conv2d(h2, d*8, scope="d_conv2d_h3", activation_fn=lrelu, batch_norm=True)
      #h4 = conv2d(h3, d*16, scope="d_conv2d_h4", activation_fn=lrelu, batch_norm=True)
      h4 = tf.reshape(h3, [batch_size, int(math.ceil(self.imsize/16.))**2*d*8])
      
      out = fully_connected(h4, 1, None, batch_norm=False, scope='d_conv2d_out')
      return out
  
  def build_model(self, batch_size):
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.X = tf.placeholder(tf.float32, [None, self.imsize * self.imsize * self.c_dim], name='real_images')
    
    # Define generator
    self.G = self.generator(self.z, batch_size)
    self.sampler = self.generator(self.z, self.sample_num, reuse_variables=True)
    # Define discriminator
    self.D_real = self.discriminator(self.X, batch_size)
    self.D_fake = self.discriminator(self.G, batch_size, reuse_variables=True)
    
    # Loss given by Jensen-Shannon Divergence
    # Discriminator Loss
    d_loss_real = tf.reduce_mean(tf.nn.
      sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.
      sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
    self.d_loss = d_loss_fake + d_loss_real
    # Generator loss
    self.g_loss = tf.reduce_mean(tf.nn
      .sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

    # Separate trainable variables for generator / discriminator
    d_vars = [t for t in tf.trainable_variables() if 'discriminator' in t.name]
    g_vars = [t for t in tf.trainable_variables() if 'generator' in t.name]
    # Define optimizer
    self.D_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.adam_beta). \
      minimize(self.d_loss, var_list=d_vars)
    self.G_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.adam_beta). \
      minimize(self.g_loss, var_list=g_vars)
  
  def train(self, max_epochs):
    t = time.time()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

      samples = sess.run(self.sampler, feed_dict={self.z: sample_z})
      save_image("generated/result0000.png", samples)
      for epoch in range(max_epochs):
        max_batch_iterations = int(self.images.num_examples / self.batch_size)
        for it in range(max_batch_iterations):
          batch = self.images.next_batch(self.batch_size)[0]

          batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
          _, dloss = sess.run([self.D_optimizer, self.d_loss],
            feed_dict={self.X: batch, self.z: batch_z})
          _, gloss = sess.run([self.G_optimizer, self.g_loss],
            feed_dict={self.z: batch_z})
        if epoch % 5 == 0:
          print("time: {:.2f}, dloss: {:.4f}, gloss: {:.4f}".format(time.time() - t, dloss, gloss))
          t = time.time()
          samples = sess.run(self.sampler, feed_dict={self.z: sample_z})
          save_image("generated/result{}.png".format(epoch), samples)



dcgan = DCGAN(None)
dcgan.train(500)
