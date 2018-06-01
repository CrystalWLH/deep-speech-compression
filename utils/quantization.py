#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:49:08 2018

@author: Samuele Garda
"""

import tensorflow as tf


def bucket_tensor(tensor,bucket_size):
  
  if not bucket_size:
    return tensor
  
  tensor = tf.reshape(tensor,[-1])
  size = tensor.get_shape().as_list()[-1]
  mul,rest = divmod(size,bucket_size)
  fill_value = tensor[-1]
  
  if mul != 0 and rest != 0:
    print("Fill : {}-{}".format(mul,rest))
    to_add = tf.ones([bucket_size-rest]) * fill_value
    tensor = tf.concat([tensor,to_add], axis =  0)
    
  if mul == 0:
    print("Original size : {}-{}".format(mul,rest))
    tensor = tf.reshape(tensor, [1,size])
    
  else:
    print("To bucket size : {}-{}".format(mul,rest))
    tensor = tf.reshape(tensor,[-1, bucket_size])
      
  return tensor


class Scaling():
  
  def __init__(self):
  
    self.tol_diff_zero = 1e-10

  
  def linear_scale(self,tensor,bucket_size):
    
    self.original_dims = tf.shape(tensor)
    self.original_length = tf.size(tensor)
    
    tensor = bucket_tensor(tensor, bucket_size)
    
    if not bucket_size:
      tensor = tf.reshape(tensor,[-1])
    
    axis = 0 if not bucket_size else 1
    
    min_rows = tf.reduce_min(tensor, axis = axis, keepdims = True)   
    max_rows = tf.reduce_max(tensor, axis = axis, keepdims = True)    
    
    alpha = max_rows - min_rows
    beta = min_rows
    
    if not bucket_size:
      alpha = tf.cond(tf.squeeze(alpha) < self.tol_diff_zero, lambda : 1.0, lambda : alpha)
      
    else:
      
      #https://www.tensorflow.org/api_docs/python/tf/scatter_nd
      
      idxs = tf.where(alpha < self.tol_diff_zero)
      n_ones = tf.shape(idxs)
      tensor = tf.scatter_nd_update(tensor, idxs, tf.ones(n_ones))
      
    self.alpha = alpha
    self.beta = beta
    
    tensor = (tensor - self.beta) / self.alpha
    
    return tensor
  
  
  def inv_linear_scale(self,tensor,bucket_size):
    
    tensor = (tensor * self.alpha) + self.beta
    
    tensor = tf.reshape(tensor[0:self.original_length], self.original_dims, name = 'quantize_weight')
    
    return tensor
  
  

def quantize_uniform(tensor,s,bucket_size,stochastic):
  
  scaling = Scaling()
  
  tensor = scaling.linear_scale(tensor,bucket_size)
  
  s = s - 1
  
  if not stochastic:
  
    tensor = tf.round(tensor * s) / s
    
  else:
    
    raise NotImplementedError("Sorry! Stochastic rounding not implemented yet!")
    
  tensor = scaling.inv_linear_scale(tensor, bucket_size)
  
  return tensor
  
  
  

def quant_conv_sequence(conv_type, inputs, filters, widths, strides,
                        dropouts, activation, data_format,batchnorm,train,
                        num_bits,bucket_size,stochastic):
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
  
  
  quantized_weights = []
  original_weights = []
  
  s = 2 ** num_bits 
  
  if conv_type == 'gated_conv':
    raise ValueError("Sorry quantized gated convolutions are not implemented!")
  else:
    conv_op = tf.nn.conv1d
  
  
  prev_layer = inputs
  
  for layer in range(len(filters)):
    layer_name = conv_type + '_layer_' + str(layer)
    with tf.variable_scope(layer_name):
      kernel,num_filters = widths[layer], filters[layer]
      
      W = tf.get_variable('W_{}'.format(str(layer)), [kernel,kernel,num_filters])
      
      original_weights.append(W)
      
      quant_W = quantize_uniform(tensor = W, s = s, bucket_size = bucket_size, stochastic = stochastic)
      
      quantized_weights.append(quant_W)
      
      conv = conv_op(value = prev_layer,filters = quant_W, stride = strides[layer], padding = 'same',
                          data_format= data_format, name = conv_type)
      
      
      if batchnorm:
        conv = tf.layers.batch_normalization(conv, axis=1 if data_format == "channels_first" else -1,
                                             training=train, name="bn")
#      else:
#        
#        bias = tf.get_variable('b_{}'.format(str(layer)), [num_filters])
#        original_weights.append(bias)
#        
#        quant_bias = quantize_uniform(tensor = bias, s = s, bucket_size = bucket_size, stochastic = stochastic)
#        quantized_weights.append(quant_bias)
#        
#        conv = tf.nn.add_bias(conv, quant_bias, data_format)
        
      conv = activation(conv)
      prev_layer = conv
      if dropouts[layer] != 0:
        prev_layer = tf.layers.dropout(prev_layer, rate=dropouts[layer], training=train, name="dropout")  
      
      tf.summary.histogram(layer_name, prev_layer)
      
  return prev_layer,quantized_weights,original_weights  


def quant_clip_and_step(optimizer, loss, quantized_weights, original_weights, clipping):
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
  
  quantized_grads = tf.gradients(loss, quantized_weights)
  
  grads, varis = quantized_grads,original_weights
        
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
      
    
 


    
    
  