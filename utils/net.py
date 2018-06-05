#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:23:23 2018

@author: Samuele Garda
"""

import tensorflow as tf

def gated_conv(inputs,filters, kernel_size, strides, activation,padding, data_format,name):
  """
  Gated convolution: Elementwise product of 1D convolved features with sigmoid activated second 1D convolution. 
  :math:`h^{i}(X) = (X * W^{i} + b^{i}) \otimes \sigma(X * V^{i} + c^{i})`
  
  :param:
    inputs (tf.Tensor) : 3D input features
    filters (int) : number of output filters
    kernel_size (int) : kernel width
    strides (int) : stride
    padding (str) : type of padding
    data_format (str) : Either `channels_first` (batch, channels, max_length) or `channels_last` (batch, max_length, channels)
    name (str) : operation name in graph
    
  :return:
    conv (tf.Tensor) : result of gated convolution operation
  """
  
  with tf.variable_scope(name):
  
    c_1 = tf.layers.conv1d(inputs = inputs, filters=filters, kernel_size= kernel_size, strides = strides, activation = activation,
                         padding = padding, data_format = data_format,name = 'conv1' )
    
    c_2 = tf.layers.conv1d(inputs = inputs, filters=filters, kernel_size = kernel_size, strides = strides, activation = activation,
                         padding = padding, data_format = data_format, name = 'conv2')
    
    conv = tf.matmul(c_1, tf.nn.sigmoid(c_2))
  
  return conv
  
  
def convolutional_sequence(conv_type, inputs, filters, widths, strides, dropouts, activation, data_format,batchnorm,train):
  """
  Apply sequence of 1D convolution operation.
  
  :param:
    conv_type (str) : type of convolution. Either `gated_conv` or `conv`
    inputs (tf.Tensor) : 3D input features
    filters (list) : sequence of filters
    widths (list) : sequence of kernel sizes
    strides (list) : sequence of strides
    activation (tf function) : activation function
    data_format (str) : Either `channels_first` (batch, channels, max_length) or `channels_last` (batch, max_length, channels) 
    batchnorm (bool) : use batch normalization
    train (bool) : wheter in train mode or not
    
  :return:
    pre_out (tf.Tensor) : result of sequence of convolutions
    
  """
  
  conv_op = gated_conv if conv_type == 'gated_conv' else tf.layers.conv1d
  
  prev_layer = inputs
  
  for layer in range(len(filters)):
    layer_name = conv_type + '_layer_' + str(layer)
    with tf.variable_scope(layer_name):
      conv = conv_op(inputs = prev_layer,filters = filters[layer],
                     kernel_size = widths[layer], strides = strides[layer],
                     activation = None if batchnorm else activation,
                     use_bias= not batchnorm,padding = 'same',
                     data_format = data_format, name = conv_type)
      
      
      if batchnorm:
        conv = tf.layers.batch_normalization(conv, axis=1 if data_format == "channels_first" else -1,
                                             training=train, name="bn")
        
      conv = activation(conv)
      prev_layer = conv
      if dropouts[layer] != 0:
        prev_layer = tf.layers.dropout(prev_layer, rate=dropouts[layer], training=train, name="dropout")  
      
      tf.summary.histogram(layer_name, prev_layer)
      
  return prev_layer

def clip_and_step(optimizer, loss, clipping):
  """
  Helper to compute/apply gradients with clipping.
  
  Parameters:
  optimizer: Subclass of tf.train.Optimizer (e.g. GradientDescent or Adam).
  loss: Scalar loss tensor.
  clipping: Threshold to use for clipping.
  
  Returns:
  The train op.
  List of gradient, variable tuples, where gradients have been clipped.
  Global norm before clipping.
  """
  grads_and_vars = optimizer.compute_gradients(loss)
  grads, varis = zip(*grads_and_vars)
        
  if clipping:
      grads, global_norm = tf.clip_by_global_norm(grads, clipping,
                                                  name="gradient_clipping")
  else:
      global_norm = tf.global_norm(grads, name="gradient_norm")
  grads_and_vars = list(zip(grads, varis))  # list call is apparently vital!!
  train_op = optimizer.apply_gradients(grads_and_vars,
                                       global_step=tf.train.get_global_step(),
                                       name="train_step")
  return train_op, grads_and_vars, global_norm    

def length(batch):
  """
  Get length of sequences in a batch of logits. Since logits are in (max_length,batch,channles) 
  they are transposed to `channels_last` format (batch, max_length, channels). If `channels_first`.
  
  :param:
    batch (tf.Tensor) : 3D input features 
    data_format (str) : Either `channels_first` (batch, channels, max_length) or `channels_last` (batch, max_length, channels) 
    
  :return:
    length (tf.Tensor) : 1D vector of sequence lengths
  """
  
  with tf.variable_scope('sequence_length'):
  
    batch = tf.transpose(batch, (1,0,2))
      
    used = tf.sign(tf.reduce_max(tf.abs(batch), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
          
  return length