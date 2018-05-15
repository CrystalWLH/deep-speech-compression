#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:01:21 2018

@author: Samuele Garda
"""

#TODO
# ADD BN?

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
  
  
def convolutional_sequence(conv_type, inputs, filters, widths, strides, activation, data_format, train):
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
                     activation = activation,padding = 'same',
                     data_format = data_format, name = conv_type)
      
      prev_layer = conv
      
      tf.summary.histogram(layer_name, prev_layer)
      
  return prev_layer

def length(batch, data_format):
  """
  Get length of sequences in a batch. Expects inputs in `channels_last` format (batch, max_length, channels). If `channels_first` 
  batch is transposed.
  
  :param:
    batch (tf.Tensor) : 3D input features 
    data_format (str) : Either `channels_first` (batch, channels, max_length) or `channels_last` (batch, max_length, channels) 
    
  :return:
    length (tf.Tensor) : 1D vector of sequence lengths
  """
  
  with tf.variable_scope("sequence_lengths"):
    
    if data_format == 'channels_first':
      batch = tf.transpose(batch, (0,2,1))
      
    used = tf.sign(tf.reduce_max(tf.abs(batch), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
          
  return length


def teacher_model_function(features, labels, mode, params):
  """
  Teacher model function. Train, predict, evaluate model.
  
  :param:
    features (tf.Tensor) : 3D input features
    labels (tf.Tensor) : 2D labels
    mode (str) : Choices (`train`,`eval`,`predict`)
    params (dict) : Parameter for the model. Should contain following keys:
      
      - data_format (str) : Either `channels_first` (batch, channels, max_length) or `channels_last` (batch, max_length, channels)
      - activation (tf function) : activation function
      - vocab_size (int) : possible output charachters
      - filters (list) : sequence of filters
      - widths (list) : sequence of kernel sizes
      - strides (list) : sequence of strides
      
  :return:
    specification (tf.estimator.EstimatorSpec)
  """
  
  with tf.variable_scope("data_format"):
  
    if params.get('data_format') == "channels_last":
      
      features = tf.transpose(features, (0, 2, 1))
      
  seqs_len = length(features, data_format = params.get('data_format'))
    
  with tf.variable_scope("model"):
    
    pre_out = convolutional_sequence(inputs = features, conv_type = params.get('conv_type'),
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            train = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.conv1d(inputs = pre_out, filters = params.get('vocab_size'), kernel_size = 1,
                                  strides= 1, activation=None,
                                  padding="same", data_format=params.get('data_format'),name="logits")
    
       
    # get logits in time major : [max_time, batch_size, num_classes]
    if params.get('data_format') == 'channels_first':
      logits = tf.transpose(logits, (2,0,1))
      
    elif params.get('data_format') == 'channels_last':
      logits = tf.transpose(logits, (1,0,2))
  
  if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL: 
    
    with tf.name_scope("predictions"):
      
      sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
        
    with tf.name_scope('predictions'):
          
      sparse_decoded = sparse_decoded[0]
      
      dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
                                              sparse_decoded.dense_shape,
                                              sparse_decoded.values)
  
      
      pred = {'decoding' : dense_decoded, 'log_prob' : log_prob, 'logits' : logits}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred)
      

  with tf.name_scope('loss'):
    
    sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
    
    batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                      inputs =  logits, 
                                      sequence_length = seqs_len)
    loss = tf.reduce_mean(batches_ctc_loss)
    tf.summary.scalar('ctc_loss',loss)
  
 
  if mode == tf.estimator.ModeKeys.TRAIN:
        
    with tf.variable_scope("optimizer"):
      train_step = tf.train.AdamOptimizer().minimize(loss,
                                                      global_step=tf.train.get_global_step())
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_step) 
  
  
  assert mode == tf.estimator.ModeKeys.EVAL

  ler = tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), sparse_labels)
  mean_ler, op = tf.metrics.mean(ler)
  
  metrics = {"ler": (mean_ler, op)}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss = loss, eval_metric_ops=metrics)




def student_model_function(features, labels, mode, params):
  """
  Teacher model function. Train, predict, evaluate model.
  
  :param:
    features (tf.Tensor) : 3D input features
    labels (tf.Tensor) : 2D labels
    mode (str) : Choices (`train`,`eval`,`predict`)
    params (dict) : Parameter for the model. Should contain following keys:
      
      - data_format (str) : Either `channels_first` (batch, channels, max_length) or `channels_last` (batch, max_length, channels)
      - activation (tf function) : activation function
      - vocab_size (int) : possible output charachters
      - filters (list) : sequence of filters
      - widths (list) : sequence of kernel sizes
      - strides (list) : sequence of strides
      
  :return:
    specification (tf.estimator.EstimatorSpec)
  """
      
  audio_features = features['audio']
  
  with tf.variable_scope('teacher_logits'):
  
    teacher_logits = tf.transpose(features['logits'],(1,0,2))
    
  with tf.variable_scope('data_format'):
    
    if params.get('data_format') == "channels_last":
      
      audio_features = tf.transpose(audio_features, (0, 2, 1))
      
  seqs_len = length(audio_features, data_format = params.get('data_format'))
        
  with tf.variable_scope("model"):
    
    pre_out = convolutional_sequence(inputs = audio_features, conv_type = params.get('conv_type'),
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            train =  True)
    
    logits = tf.layers.conv1d(inputs = pre_out, filters = params.get('vocab_size'), kernel_size = 1,
                                  strides= 1, activation=None,
                                  padding="same", data_format=params.get('data_format'),name="logits")
    
    
    if params.get('data_format') == 'channels_first':
      trans_logits = tf.transpose(logits, (2,0,1))
        
    elif params.get('data_format') == 'channels_last':
      trans_logits = tf.transpose(logits, (1,0,2))
  
  student_logits_shape = tf.shape(trans_logits)
  teacher_logits_shape = tf.shape(teacher_logits)
  
  tf.assert_equal(student_logits_shape,teacher_logits_shape)
        
  if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL: 
    
    with tf.name_scope("predictions"):
      
      sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder (logits, seqs_len)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
        
    with tf.name_scope('predictions'):
          
      sparse_decoded = sparse_decoded[0]
      
      dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
                                              sparse_decoded.dense_shape,
                                              sparse_decoded.values)
  
      
      pred = {'decoding' : dense_decoded, 'log_prob' : log_prob}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred)
  
  with tf.name_scope('loss'):
    
    with tf.variable_scope('ctc_loss'):
    
      sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
      
      batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                        inputs =  logits, 
                                        sequence_length = seqs_len)
      ctc_loss = tf.reduce_mean(batches_ctc_loss)
      tf.summary.scalar('ctc_loss',ctc_loss)
      
    with tf.variable_scope('distillation_loss'):
      
      # add softmax on right axis?
      soft_targets = teacher_logits/params.get('temperature')
      
      logits_fl = tf.reshape(logits, [tf.shape(logits)[1],-1])
      st_fl = tf.reshape(soft_targets,[tf.shape(logits)[1],-1])
                  
      xent_soft_targets = tf.reduce_mean(-tf.reduce_sum(st_fl * tf.log(logits_fl), axis=1))
      
      tf.summary.scalar('soft_target_xent', xent_soft_targets)
      
      
    loss = ctc_loss +  xent_soft_targets
  
 
  if mode == tf.estimator.ModeKeys.TRAIN:
        
    with tf.variable_scope("optimizer"):
      train_step = tf.train.AdamOptimizer().minimize(loss,
                                                      global_step=tf.train.get_global_step())
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_step) 
  
  
  assert mode == tf.estimator.ModeKeys.EVAL

  ler = tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), sparse_labels)
  mean_ler, op = tf.metrics.mean(ler)
  
  metrics = {"ler": (mean_ler, op)}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss = loss, eval_metric_ops=metrics)
            
    
    
    
  
    
    
    
  
  
