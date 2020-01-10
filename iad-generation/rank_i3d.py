

import tensorflow as tf
import numpy as np

import os, cv2

import PIL.Image as Image

import sonnet as snt



@tf.custom_gradient
def rank_layer(x):
  def grad(dy):

    global rank_out

    #calculate the rank
    print("x.get_shape():", x.get_shape())
    print("dy.get_shape():", dy.get_shape())
    ranks = tf.math.multiply(x, dy)
    print("ranks.get_shape():", ranks.get_shape())
    ranks = tf.reduce_sum(ranks, axis=(0, 1, 2, 3)) #combine spatial and temporal points together

    print("ranks.get_shape():", ranks.get_shape())

    #normalize the rank by the input size
    norm_term = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), tf.float32)
    ranks = tf.math.divide(ranks, norm_term) 

    rank_out.append(ranks)

    return dy
  return tf.identity(x), grad

class Unit3D(snt.AbstractModule):
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, output_channels,
               kernel_shape=(1, 1, 1),
               stride=(1, 1, 1),
               activation_fn=tf.nn.relu,
               use_batch_norm=True,
               use_bias=False,
               name='unit_3d'):
    """Initializes Unit3D module."""
    super(Unit3D, self).__init__(name=name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._use_batch_norm = use_batch_norm
    self._activation_fn = activation_fn
    self._use_bias = use_bias

  def _build(self, inputs, is_training):
    """Connects the module to inputs.

    Args:
      inputs: Inputs to the Unit3D component.
      is_training: whether to use training mode for snt.BatchNorm (boolean).

    Returns:
      Outputs from the module.
    """
    net = snt.Conv3D(output_channels=self._output_channels,
                     kernel_shape=self._kernel_shape,
                     stride=self._stride,
                     padding=snt.SAME,
                     use_bias=self._use_bias)(inputs)
    if self._use_batch_norm:
      bn = snt.BatchNorm()
      net = bn(net, is_training=is_training, test_local_stats=False)
    if self._activation_fn is not None:
      net = self._activation_fn(net)
    return net


class InceptionI3d(snt.AbstractModule):
  """Inception-v1 I3D architecture.

  The model is introduced in:

    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    Joao Carreira, Andrew Zisserman
    https://arxiv.org/pdf/1705.07750v1.pdf.

  See also the Inception architecture, introduced in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.
  """

  # Endpoints of the model in order. During construction, all the endpoints up
  # to a designated `final_endpoint` are returned in a dictionary as the
  # second return value.
  VALID_ENDPOINTS = (
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions',
  )

  def __init__(self, num_classes=400, spatial_squeeze=True,
               final_endpoint='Logits', name='inception_i3d'):
    """Initializes I3D model instance.

    Args:
      num_classes: The number of outputs in the logit layer (default 400, which
          matches the Kinetics dataset).
      spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
          before returning (default True).
      final_endpoint: The model contains many possible endpoints.
          `final_endpoint` specifies the last endpoint for the model to be built
          up to. In addition to the output at `final_endpoint`, all the outputs
          at endpoints up to `final_endpoint` will also be returned, in a
          dictionary. `final_endpoint` must be one of
          InceptionI3d.VALID_ENDPOINTS (default 'Logits').
      name: A string (optional). The name of this module.

    Raises:
      ValueError: if `final_endpoint` is not recognized.
    """

    if final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % final_endpoint)

    super(InceptionI3d, self).__init__(name=name)
    self._num_classes = num_classes
    self._spatial_squeeze = spatial_squeeze
    self._final_endpoint = final_endpoint

    self.rank_out = []

  def _build(self, inputs, is_training, dropout_keep_prob=1.0):
    """Connects the model to inputs.

    Args:
      inputs: Inputs to the model, which should have dimensions
          `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
      is_training: whether to use training mode for snt.BatchNorm (boolean).
      dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
          [0, 1)).

    Returns:
      A tuple consisting of:
        1. Network output at location `self._final_endpoint`.
        2. Dictionary containing all endpoints up to `self._final_endpoint`,
           indexed by endpoint name.

    Raises:
      ValueError: if `self._final_endpoint` is not recognized.
    """

    target_layers = []

    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

    net = inputs
    end_points = {}
    end_point = 'Conv3d_1a_7x7'
    net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
                 stride=[2, 2, 2], name=end_point)(net, is_training=is_training)
    end_points[end_point] = net

    target_layers.append(net)
    net = rank_layer(net)

    if self._final_endpoint == end_point: return net, end_points, target_layers
    end_point = 'MaxPool3d_2a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers
    end_point = 'Conv3d_2b_1x1'
    net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                 name=end_point)(net, is_training=is_training)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers
    end_point = 'Conv3d_2c_3x3'
    net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                 name=end_point)(net, is_training=is_training)
    end_points[end_point] = net

    target_layers.append(net)
    net = rank_layer(net)

    if self._final_endpoint == end_point: return net, end_points, target_layers
    end_point = 'MaxPool3d_3a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_3b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_3c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net

    target_layers.append(net)
    net = rank_layer(net)

    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'MaxPool3d_4a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_4b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_4c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_4d'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_4e'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_4f'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net

    target_layers.append(net)
    net = rank_layer(net)

    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'MaxPool3d_5a_2x2'
    net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_5b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0a_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Mixed_5c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net

    target_layers.append(net)
    net = rank_layer(net)

    if self._final_endpoint == end_point: return net, end_points, target_layers

    end_point = 'Logits'
    with tf.variable_scope(end_point):
      net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)
      net = tf.nn.dropout(net, dropout_keep_prob)
      logits = Unit3D(output_channels=self._num_classes,
                      kernel_shape=[1, 1, 1],
                      activation_fn=None,
                      use_batch_norm=False,
                      use_bias=True,
                      name='Conv3d_0c_1x1')(net, is_training=is_training)
      if self._spatial_squeeze:
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
    averaged_logits = tf.reduce_mean(logits, axis=1)
    end_points[end_point] = averaged_logits
    if self._final_endpoint == end_point: return averaged_logits, end_points, target_layers

    end_point = 'Predictions'
    predictions = tf.nn.softmax(averaged_logits)
    end_points[end_point] = predictions
    return predictions, end_points, target_layers

def generate_activation_map(input_ph, isRGB):

  global rank_out
  rank_out = []


  scope_name = 'RGB' if isRGB else 'Flow'


  is_training = tf.placeholder_with_default(False, shape=(), name="is_training_ph")
  with tf.variable_scope(scope_name):
    logits, _, target_layers = InceptionI3d( num_classes=101,
        spatial_squeeze=True,
        final_endpoint='Logits')(input_ph, is_training)

  gradients = tf.gradients(logits, input_ph)
  rank_out = rank_out[::-1]

  return target_layers, rank_out

###################
# FILE IO         #
###################

def obtain_files(directory_file):
  # read a *.list file and get the file names and labels
  ifile = open(directory_file, 'r')
  line = ifile.readline()
  
  filenames, labels, max_length = [],[],0
  while len(line) != 0:

    line = line.split()

    filename, label = line
    
    filenames.append(filename)
    labels.append(label)
    file_length = len(os.listdir(filename))-1

    if(file_length > max_length):
      max_length = file_length

    line = ifile.readline()

  return zip(filenames, labels), max_length

def read_file(file, input_placeholder, isRGB):
  if(isRGB):
    return read_file_frames(file, input_placeholder)
  return read_file_flow(file, input_placeholder)

def read_file_frames(file, input_placeholder):
  # read a file and concatenate all of the frames
  # pad or prune the video to the given length

  try:
    # input_shape
    _, num_frames, h, w, ch = input_placeholder.get_shape()

    # read available frames in file
    img_data = []
    for r, d, f in os.walk(file):
      #print(file, r)
      #print("num_files:", len(f))

      limit = min(int(num_frames), len(f))
      
      for i in range(1, limit+1):
        filename = os.path.join(r, 'image_{0}.jpg'.format( str(i).zfill(5) ))
        try:
          img = Image.open(filename)
        except:
          print("Cannot open file named: "+ filename)

        # resize and crop to fit input size
        if(img.width > img.height):
          img = np.array(cv2.resize(np.array(img),(int((256.0/img.height) * img.width+1), 256))).astype(np.float32)
        else:
          img = np.array(cv2.resize(np.array(img),(256, int((256.0/img.width) * img.height+1)))).astype(np.float32)
    
        crop_x = int((img.shape[0] - h)/2)
        crop_y = int((img.shape[1] - w)/2)
        img = img[crop_x:crop_x+w, crop_y:crop_y+h,:] 

        img_data.append(np.array(img))

    img_data = np.array(img_data).astype(np.float32)

    # pad file to appropriate length
    buffer_len = int(num_frames) - len(img_data)
    img_data = np.pad(np.array(img_data), 
          ((0,buffer_len), (0,0), (0,0),(0,0)), 
          'constant', 
          constant_values=(0,0))
    length_ratio = float(limit) / len(img_data)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data, length_ratio 
  except:
    print("Cannot open file: "+ file)

def read_file_flow(file, input_placeholder):
  # read a file and concatenate all of the frames
  # pad or prune the video to the given length

  try:
    # input_shape
    _, num_frames, h, w, ch = input_placeholder.get_shape()

    # read available frames in file
    img_data = []
    for r, d, f in os.walk(file):

      limit = min(int(num_frames), len(f)/2)
      #print('limit:', limit)

      for i in range(1, limit+1):
        img_full = []
        for ax in ['x', 'y']:

          filename = os.path.join(r, 'flow_{0}_{1}.jpg'.format( ax, str(i).zfill(5) ))
          try:
            img = Image.open(filename)
          except:
            print("Cannot open file named: "+ filename)

          #print("t0")
          # resize and crop to fit input size
          if(img.width > img.height):
            img = np.array(cv2.resize(np.array(img),(int((256.0/img.height) * img.width+1), 256))).astype(np.float32)
          else:
            img = np.array(cv2.resize(np.array(img),(256, int((256.0/img.width) * img.height+1)))).astype(np.float32)
      
          #print("t1: ", img.shape, h, w)
          crop_x = int((img.shape[0] - h)/2)
          #print("t1.5: ", img.shape, h, w)
          crop_y = int((img.shape[1] - w)/2)
          #print("t1.75: ", img.shape, h, w)
          img_full.append( img[crop_x:crop_x+w, crop_y:crop_y+h] )
          #print("t2")

        #print("img_full:", len(img_full) )
        img_data.append(np.array(img_full))

    img_data = np.array(img_data).astype(np.float32).transpose([0, 2, 3, 1])
    #print("img_data_shape:", img_data.shape)

    # pad file to appropriate length
    buffer_len = int(num_frames) - len(img_data)
    img_data = np.pad(np.array(img_data), 
          ((0,buffer_len), (0,0), (0,0),(0,0)), 
          'constant', 
          constant_values=(0,0))
    length_ratio = float(limit) / len(img_data)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data, length_ratio 
  except:
    print("Cannot open file directory: "+ file)


###################
# MODEL FUNCTIONS #
###################

#import i3d

INPUT_DATA_SIZE = {"t": 64, "h":224, "w":224, "c":3}
CNN_FEATURE_COUNT = [64, 192, 480, 832, 1024]

def get_input_placeholder(isRGB, batch_size, num_frames=INPUT_DATA_SIZE["t"]):
  # returns a placeholder for the C3D input
  num_channels = INPUT_DATA_SIZE["c"] if isRGB else 2

  return tf.placeholder(tf.float32, 
      shape=(batch_size, num_frames, INPUT_DATA_SIZE["h"], INPUT_DATA_SIZE["w"], num_channels),
      name="c3d_input_ph")


def get_output_placeholder(batch_size):
  # returns a placeholder for the C3D output (currently unused)
  return tf.placeholder(tf.int32, 
      shape=(batch_size),
      name="c3d_label_ph")


def get_variables(isRGB):
  '''Define all of the variables for the convolutional layers of the C3D model. 
  We ommit the FC layers as these layers are used to perform reasoning and do 
  not contain feature information '''
  scope_name = 'RGB' if isRGB else 'Flow'

  variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == scope_name and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
      variable_map[variable.name.replace(':0', '')] = variable

  return variable_map

def load_model(input_ph, isRGB):
  activation_maps, rankings = generate_activation_map(input_ph, isRGB)

  for v in rankings:
    print("v: ", v.get_shape())

  variable_name_list = get_variables(isRGB)
  saver = tf.train.Saver(variable_name_list.values(), reshape=True)

  return activation_maps, rankings, saver


