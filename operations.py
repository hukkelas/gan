import tensorflow as tf

def batch_norm_and_activation(inputs, batch_norm, activation_fn):
  out = inputs
  if batch_norm:
    out = tf.contrib.layers.batch_norm(out)
  if activation_fn:
    out = activation_fn(out)
  return out   

def fully_connected(inputs, output_dim, activation_fn=None, batch_norm=False, scope=None, stddev=0.02):
  shape = inputs.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    weights = tf.get_variable("weights", [shape[1], output_dim],
      dtype=tf.float32,
      initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable("bias", shape=[output_dim])

    out = tf.matmul(inputs, weights) + biases
    return batch_norm_and_activation(out, batch_norm, activation_fn)


def deconv2d(inputs, output_shape, kernel=5, strides=[1,2,2,1], scope=None, stddev=0.02, activation_fn=tf.nn.relu, batch_norm=True):
  input_shape = inputs.get_shape().as_list()
  with tf.variable_scope(scope or "deconv2d"):
    # Get filter/kernel: [height, width, output_channels, input_channels]
    weights = tf.get_variable('weights', [kernel, kernel, output_shape[-1], input_shape[-1]],
      initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

    out = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=strides) + biases
    return batch_norm_and_activation(out, batch_norm, activation_fn)

def conv2d(inputs, output_dim, kernel=5, strides=[1,2,2,1], scope=None, stddev=0.02, activation_fn=None, batch_norm=False):
  input_shape = inputs.get_shape().as_list()
  with tf.variable_scope(scope or "conv2d"):
    weights = tf.get_variable('weights', [kernel, kernel, input_shape[-1], output_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
    
    out = tf.nn.conv2d(inputs, weights, strides, padding='SAME') + biases
    return batch_norm_and_activation(out, batch_norm, activation_fn)

def lrelu(input, alpha=0.2):
  return tf.maximum(input, input*0.2)



