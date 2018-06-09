#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:01:21 2018

@author: Samuele Garda
"""

import tensorflow as tf
from utils.net import convolutional_sequence,length,clip_and_step
from utils.quantization import quant_conv_sequence,quant_clip_and_step

 
def teacher_model_function(features, labels, mode, params):
  """
  Train and deploy teacher model.
  
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
      - dropouts (list) : sequence of dropouts values
      - bn (bool) : use batch normalization
      - adam_lr (float) : adam learning rate
      - adam_eps (float) : adam epsilon
      - clipping (int) : clipping threshold
      
  :return:
    specification (tf.estimator.EstimatorSpec)
  """
  
  with tf.variable_scope("data_format"):
  
    if params.get('data_format') == "channels_last":
      
      features = tf.transpose(features, (0, 2, 1))
    
  with tf.variable_scope("model"):
    
    pre_out = convolutional_sequence(inputs = features, conv_type = params.get('conv_type'),
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            dropouts = params.get('dropouts'),
                            batchnorm = params.get('bn'),
                            train = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.conv1d(inputs = pre_out, filters = params.get('vocab_size'), kernel_size = 1,
                                  strides= 1, activation=None,
                                  padding="same", data_format=params.get('data_format'),name="logits")
    
       
    # get logits in time major : [max_time, batch_size, num_classes]
    if params.get('data_format') == 'channels_first':
      logits = tf.transpose(logits, (2,0,1))
      
    elif params.get('data_format') == 'channels_last':
      logits = tf.transpose(logits, (1,0,2))
    
  seqs_len = length(logits)
  
  with tf.name_scope('decoder'):
    sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
        
    with tf.name_scope('predictions'):
          
      sparse_decoded = sparse_decoded[0]
      
      dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
                                              sparse_decoded.dense_shape,
                                              sparse_decoded.values)
  
      pred = {'decoding' : dense_decoded, 'log_prob' : log_prob, 'logits' : logits}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred)
  
    
  sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
      
  ler = tf.reduce_mean(tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), sparse_labels), name = 'ler')
    
  tf.summary.scalar('ler',ler)
      

  with tf.name_scope('loss'):
    
    batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                      inputs =  logits, 
                                      sequence_length = seqs_len)
    loss = tf.reduce_mean(batches_ctc_loss)
    tf.summary.scalar('ctc_loss',loss)
  
 
  if mode == tf.estimator.ModeKeys.TRAIN:
    
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = params.get('adam_lr'), epsilon = params.get('adam_eps'))
      if params.get('bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, params.get('clipping'))
      else:
        train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, params.get('clipping'))
      
    with tf.name_scope("visualization"):
      for g, v in grads_and_vars:
        if v.name.find("kernel") >= 0:
          tf.summary.scalar(v.name.replace(':0','_') + "gradient_norm", tf.norm(g))
      tf.summary.scalar("global_gradient_norm", glob_grad_norm)
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op) 
  
  
  assert mode == tf.estimator.ModeKeys.EVAL
  
  mean_ler, op = tf.metrics.mean(ler)
  
  metrics = {"ler": (mean_ler, op)}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss = loss, eval_metric_ops=metrics)


def student_model_function(features, labels, mode, params):
  """
  Train and deploy student model.
  
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
      - dropouts (list) : sequence of dropouts values
      - bn (bool) : use batch normalization
      - adam_lr (float) : adam learning rate
      - adam_eps (float) : adam epsilon
      - clipping (int) : clipping threshold
      - temperature (int) : distillation temperature
      - alpha (float) : parameters loss weighted average
      
  :return:
    specification (tf.estimator.EstimatorSpec)
  """
      
  audio_features = features['audio']
  
  with tf.variable_scope('teacher_logits'):
  
    teacher_logits = tf.transpose(features['logits'],(1,0,2))
            
  with tf.variable_scope('data_format'):
    
    if params.get('data_format') == "channels_last":
      
      audio_features = tf.transpose(audio_features, (0, 2, 1))
        
  with tf.variable_scope("model"):
    
    pre_out = convolutional_sequence(inputs = audio_features, conv_type = params.get('conv_type'),
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            dropouts = params.get('dropouts'),
                            batchnorm = params.get('bn'),
                            train = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.conv1d(inputs = pre_out, filters = params.get('vocab_size'), kernel_size = 1,
                                  strides= 1, activation=None,
                                  padding="same", data_format=params.get('data_format'),name="logits")
    
    
     # get logits in time major : [max_time, batch_size, num_classes]
    if params.get('data_format') == 'channels_first':
      logits = tf.transpose(logits, (2,0,1))
        
    elif params.get('data_format') == 'channels_last':
      logits = tf.transpose(logits, (1,0,2))
  
  seqs_len = length(logits)
  
  with tf.name_scope('decoder'):
    
    sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)        
  
  if mode == tf.estimator.ModeKeys.PREDICT:
        
    with tf.name_scope('predictions'):
          
      sparse_decoded = sparse_decoded[0]
      
      dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
                                              sparse_decoded.dense_shape,
                                              sparse_decoded.values)
  
      
      pred = {'decoding' : dense_decoded, 'log_prob' : log_prob}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred)
  
 
    
  sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)

  ler = tf.reduce_mean(tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), sparse_labels))
    
  tf.summary.scalar('ler',ler)

    
  with tf.name_scope('ctc_loss'):
      
    batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                      inputs =  logits, 
                                      sequence_length = seqs_len)
    
    ctc_loss =  tf.reduce_mean(batches_ctc_loss)
    
    tf.summary.scalar('ctc_loss',ctc_loss)
      
  with tf.name_scope('distillation_loss'):
    
    temperature = params.get('temperature')
    
    soft_targets = tf.nn.softmax(teacher_logits / temperature)
    soft_logits = tf.nn.softmax(logits / temperature)
    
    logits_fl = tf.reshape(soft_logits, [tf.shape(logits)[1],-1])
    st_fl = tf.reshape(soft_targets,[tf.shape(logits)[1],-1])
    
    tf.assert_equal(tf.shape(logits_fl),tf.shape(st_fl))
    
    xent_soft_targets = tf.reduce_mean(-tf.reduce_sum(st_fl * tf.log(logits_fl), axis=1))
    
    tf.summary.scalar('st_xent', xent_soft_targets)
    

  with tf.name_scope('total_loss'):
    alpha = params.get('alpha')
    sq_temper = tf.cast(tf.square(temperature),tf.float32)
    
#   "Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2
#   it is important to multiply them by T^2 when using both hard and soft targets"  
    loss =  ((1 - alpha) * ctc_loss)  +  (alpha * xent_soft_targets  * sq_temper)
    tf.summary.scalar('total_loss', loss)
    
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = params.get('adam_lr'), epsilon = params.get('adam_eps'))
      if params.get('bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, params.get('clipping'))
      else:
        train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, params.get('clipping'))
        
    with tf.name_scope("visualization"):
      for g, v in grads_and_vars:
        if v.name.find("kernel") >= 0:
          tf.summary.scalar(v.name.replace(':0','_') + "gradient_norm", tf.norm(g))
      tf.summary.scalar("global_gradient_norm", glob_grad_norm)
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op) 
  
  
  assert mode == tf.estimator.ModeKeys.EVAL

  mean_ler, op = tf.metrics.mean(ler)
  
  metrics = {"ler": (mean_ler, op)}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss = loss, eval_metric_ops=metrics)



def quant_student_model_function(features, labels, mode, params):
  """
  Train and deploy student network with trained with quantized distillation. 
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
      - dropouts (list) : sequence of dropouts values
      - bn (bool) : use batch normalization
      - adam_lr (float) : adam learning rate
      - adam_eps (float) : adam epsilon
      - clipping (int) : clipping threshold
      - temperature (int) : distillation temperature
      - alpha (float) : parameters loss weighted average
      - num_bits (int) : number of bits for quantizing weights
      - bucket_size(int) : size of buckets for weights
      - stochastic (bool) : use stochastic rounding in quantization
      
  :return:
    specification (tf.estimator.EstimatorSpec)
  """
      
  audio_features = features['audio']
  
  with tf.variable_scope('teacher_logits'):
  
    teacher_logits = tf.transpose(features['logits'],(1,0,2))
            
  with tf.variable_scope('data_format'):
    
    if params.get('data_format') == "channels_last":
      
      audio_features = tf.transpose(audio_features, (0, 2, 1))
        
  with tf.variable_scope("model"):
    
    logits,quant_weights,original_weights = quant_conv_sequence(inputs = audio_features,
                            conv_type = params.get('conv_type'),
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            dropouts = params.get('dropouts'),
                            batchnorm = params.get('bn'),
                            train = mode == tf.estimator.ModeKeys.TRAIN,
                            vocab_size = params.get('vocab_size'),
                            num_bits = params.get('num_bits'),
                            bucket_size = params.get('bucket_size'),
                            stochastic = params.get('stochastic'),
                            quant_last_layer = params.get('quant_last_layer'))
    
    
     # get logits in time major : [max_time, batch_size, num_classes]
    if params.get('data_format') == 'channels_first':
      logits = tf.transpose(logits, (2,0,1))
        
    elif params.get('data_format') == 'channels_last':
      logits = tf.transpose(logits, (1,0,2))
  
  seqs_len = length(logits)
  
  with tf.name_scope('decoder'):
    
    sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)        
  
  if mode == tf.estimator.ModeKeys.PREDICT:
        
    with tf.name_scope('predictions'):
          
      sparse_decoded = sparse_decoded[0]
      
      dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
                                              sparse_decoded.dense_shape,
                                              sparse_decoded.values)
  
      
      pred = {'decoding' : dense_decoded, 'log_prob' : log_prob}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred)
  
  
  sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
  
  ler = tf.reduce_mean(tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), sparse_labels))
    
  tf.summary.scalar('ler',ler)
    
  with tf.name_scope('ctc_loss'):
      
    batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                      inputs =  logits, 
                                      sequence_length = seqs_len)
    
    ctc_loss =  tf.reduce_mean(batches_ctc_loss)
    
    tf.summary.scalar('ctc_loss',ctc_loss)
      
  with tf.name_scope('distillation_loss'):
    
    temperature = params.get('temperature')
    
    soft_targets = tf.nn.softmax(teacher_logits / temperature)
    soft_logits = tf.nn.softmax(logits / temperature)
    
    logits_fl = tf.reshape(soft_logits, [tf.shape(logits)[1],-1])
    st_fl = tf.reshape(soft_targets,[tf.shape(logits)[1],-1])
    
    tf.assert_equal(tf.shape(logits_fl),tf.shape(st_fl))
    
    xent_soft_targets = tf.reduce_mean(-tf.reduce_sum(st_fl * tf.log(logits_fl), axis=1))
    
    tf.summary.scalar('st_xent', xent_soft_targets)
    

  with tf.name_scope('total_loss'):
    alpha = params.get('alpha')
    sq_temper = tf.cast(tf.square(temperature),tf.float32)
    
#   "Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2
#   it is important to multiply them by T^2 when using both hard and soft targets"  
    loss =  ((1 - alpha) * ctc_loss)  +  ((alpha * xent_soft_targets))  * sq_temper
    tf.summary.scalar('total_loss', loss)
    
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = params.get('adam_lr'), epsilon = params.get('adam_eps'))
      if params.get('bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, quant_global_norm, orig_global_norm,orig_grads,quant_grads = quant_clip_and_step(optimizer, loss, params.get('clipping'), quant_weights, original_weights)
      else:
        train_op, quant_global_norm, orig_global_norm,orig_grads,quant_grads = quant_clip_and_step(optimizer, loss, params.get('clipping'), quant_weights, original_weights)
        
    with tf.name_scope("visualization"):
      for idx,(g_orig, g_quant) in enumerate(zip(orig_grads,quant_grads)):
        tf.summary.scalar("model/conv_layer_{}/conv/kernel_gradient_norm".format(idx), tf.norm(g_orig))
        tf.summary.scalar("model/quant_conv_layer_{}/conv/kernel_gradient_norm".format(idx), tf.norm(g_quant))
  
      tf.summary.scalar("global_gradient_norm", orig_global_norm)
      tf.summary.scalar("quant_global_gradient_norm", quant_global_norm)
      
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op) 
  
  
  assert mode == tf.estimator.ModeKeys.EVAL

  mean_ler, op = tf.metrics.mean(ler)
  
  metrics = {"ler": (mean_ler, op)}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss = loss, eval_metric_ops=metrics)
