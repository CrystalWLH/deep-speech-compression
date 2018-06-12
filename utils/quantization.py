#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:49:08 2018

@author: Samuele Garda
"""

import tensorflow as tf


def bucket_tensor(tensor,bucket_size):
  """
  Create reshape tensor in tensors of size `bucket_size`.
  Avoid magnitude imbalance when scaling.
  
  :param:
    tensor (tf.Tensor) : tensor 
    
    bucket_size (int) : size of buckets for reshaping
  
  :return:
    tensor (tf.Tensor) : bucketed tensor
  """
  
  if not bucket_size:
    return tensor
  
  tensor = tf.reshape(tensor,[-1])
  size = tensor.get_shape().as_list()[-1]
  mul,rest = divmod(size,bucket_size)
  fill_value = tensor[-1]
  
  if mul != 0 and rest != 0:
    to_add = tf.ones([bucket_size-rest]) * fill_value
    tensor = tf.concat([tensor,to_add], axis =  0)
    
  if mul == 0:
    tensor = tf.reshape(tensor, [1,size])
    
  else:
    tensor = tf.reshape(tensor,[-1, bucket_size])
      
  return tensor


class Scaling():
  """
  Scaler object.
  """
  
  def __init__(self):
    """
    Construct new object
    """
    self.tol_diff_zero = 1e-10

  
  def linear_scale(self,tensor,bucket_size):
    """
    Scale linearly tensor. Firt tensor is bucketed.
    
    :param:
      tensor (tf.Tensor) : tensor 
      
      bucket_size (int) : size of buckets for reshaping
    
    :return:
      tensor (tf.Tensor) : bucketed tensor
    
    :math:`sc(v)= \\frac{v - \\beta}{\\alpha}`
    
    where:
      :math:`\\alpha = max_{i}v_{i} - min_{i}v_{i}`,
      :math:`\\beta = min_{i}v_{i}`
    """
    
    self.original_dims = tf.shape(tensor)
    self.original_length = tf.size(tensor)
    
    with tf.variable_scope("scaling"):
    
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
        below_idxs = tf.where(alpha <= self.tol_diff_zero)
        below_updates = tf.ones(tf.size(below_idxs))
        
        above_idxs = tf.where(alpha > self.tol_diff_zero)
        above_updates = tf.gather_nd(alpha,above_idxs)
        
        indices = tf.concat([below_idxs,above_idxs],axis = 0)
        updates = tf.concat([below_updates,above_updates], axis = 0)
         
        alpha = tf.scatter_nd(indices, updates, tf.shape(alpha, out_type = tf.int64))
        
      self.alpha = alpha
      self.beta = beta
      
      tensor = (tensor - self.beta) / self.alpha
    
    return tensor
  
  
  def inv_linear_scale(self,tensor):
    """
    Inverse of scaling function.
    
    :param:
      tensor (tf.Tensor) : tensor 
      
      bucket_size (int) : size of buckets for reshaping
    
    :return:
      tensor (tf.Tensor) : original tensor
    """
    
    tensor = (tensor * self.alpha) + self.beta
    
    tensor = tf.reshape(tensor,[-1])[0:self.original_length]
    
    tensor = tf.reshape(tensor, self.original_dims)
    
    return tensor
  
  

def quantize_uniform(tensor,s,bucket_size,stochastic):
  
  with tf.variable_scope("quantize_weights"):
  
    scaling = Scaling()
    
    tensor = scaling.linear_scale(tensor,bucket_size)
    
    s = s - 1
    
    if not stochastic:
    
      tensor = tf.round(tensor * s) / s
      
    else:
      
      l_vector = tf.floor(tensor * s)
      probabilities = s * tensor - l_vector
      tensor = l_vector / s 
      
      curr_rand = tf.random_uniform(tf.shape(tensor))
      curr_rand = tf.cast(curr_rand <= probabilities, tf.float32)
      tensor = tensor + curr_rand * 1/s
      
    tensor = scaling.inv_linear_scale(tensor)
  
  return tensor
  
  
  

def quant_conv_sequence(conv_type, inputs, filters, widths, strides,
                        dropouts, activation, data_format,batchnorm,train,
                        num_bits,bucket_size,stochastic,vocab_size,quant_last_layer):
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
    vocab_size (int) : number of classes in logits
    num_bits (int) : number of bits for quantizing weights
    bucket_size(int) : size of buckets for weights
    stochastic (bool) : use stochastic rounding in quantization
    quant_last_layer (bool) : quantize weights of last layer
    
  :return:
    pre_out (tf.Tensor) : result of sequence of convolutions
    
  """
  
  map_data_format = {'channels_first' : 'NCW', 'channels_last' : 'NWC' }
  
  quantized_weights = []
  original_weights = []
  
  s = 2 ** num_bits 
  
  if conv_type == 'gated_conv':
    raise NotImplementedError("Sorry quantized gated convolutions are not implemented yet!")
  else:
    conv_op = tf.nn.conv1d
  
  channels_axis = 1 if data_format == "NCW" else -1
  
  prev_layer = inputs
  
  for layer in range(len(filters)):
    layer_name = conv_type + '_layer_' + str(layer)
    with tf.variable_scope(layer_name,reuse=tf.AUTO_REUSE):
      
      in_channels = prev_layer.get_shape().as_list()[channels_axis]
      
      kernel,num_filters = widths[layer], filters[layer]
      
      W = tf.get_variable('quant_kernel_{}'.format(str(layer)), [kernel,in_channels,num_filters])
      
      original_weights.append(W)
      
      quant_W = quantize_uniform(tensor = W, s = s, bucket_size = bucket_size, stochastic = stochastic)
      
      quantized_weights.append(quant_W)
      
      conv = conv_op(value = prev_layer,filters = quant_W, stride = strides[layer], padding = 'SAME',
                          data_format= map_data_format.get(data_format), name = conv_type)
      
      
      if batchnorm:
        conv = tf.layers.batch_normalization(conv, axis=1 if data_format == "NCW" else -1,
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
      
    
  with tf.variable_scope('logits',reuse=tf.AUTO_REUSE):
    
    in_channels = prev_layer.get_shape().as_list()[channels_axis]
    
    W_logits = tf.get_variable('logits', [1,in_channels,vocab_size])
    
    original_weights.append(W_logits)
    
    quantized_weights.append(W_logits)
    
    if quant_last_layer:
    
      W_logits = quantize_uniform(tensor = W, s = s, bucket_size = bucket_size, stochastic = stochastic)
    
      quantized_weights[-1] = W_logits
    
    logits = conv_op(value = prev_layer,filters = W_logits, stride = 1, padding = 'SAME',
                        data_format= map_data_format.get(data_format), name = 'logits')
      
  return logits,quantized_weights,original_weights  


def quant_clip_and_step(optimizer, loss,clipping,quantized_weights, original_weights):
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
  
  quantized_grads = tf.gradients(loss, quantized_weights,name = 'quant_grads')
  
  original_grads = tf.gradients(loss, original_weights, name = 'orig_grads')
  
  grads, varis = quantized_grads,original_weights
        
  if clipping:
      grads, quant_global_norm = tf.clip_by_global_norm(grads, clipping,
                                                  name="gradient_clipping")
  
  else:
      quant_global_norm = tf.global_norm(grads, name="gradient_norm")
      
  original_global_norm = tf.global_norm(original_grads, name="original_gradient_norm")
      
  grads_and_vars = list(zip(grads, varis))  # list call is apparently vital!!
  
  train_op = optimizer.apply_gradients(grads_and_vars,
                                       global_step=tf.train.get_global_step(),
                                       name="train_step")
  
  return train_op, quant_global_norm, original_global_norm,original_grads,quantized_grads  
      
    
if __name__ == "__main__":
  
  features = tf.placeholder(tf.float32,[64,350,39])
  
  pre_out = quant_conv_sequence(inputs = features, conv_type = 'quant_conv',filters = [32,32],widths = [7,7],strides = [1,1],
                                activation = tf.nn.relu,data_format = "NWC",dropouts = [0,0],batchnorm = False,
                                train = False,num_bits = 4, bucket_size = 256,stochastic = True, quant_last_layer = False, vocab_size = 28)
  
  print(pre_out)

    
    
  