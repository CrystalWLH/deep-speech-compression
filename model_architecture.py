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
  
  with tf.variable_scope('gated_conv_' + name):
  
    c_1 = tf.layers.conv1d(inputs = inputs, filters=filters, kernel_size= kernel_size, strides = strides, activation = activation,
                         padding = padding, data_format = data_format)
    
    c_2 = tf.layers.conv1d(inputs = inputs, filters=filters, kernel_size = kernel_size, strides = strides, activation = activation,
                         padding = padding, data_format = data_format)
    
    conv = tf.matul(c_1, tf.nn.sigmoid(c_2))
  
  return conv
  
  
def convolutional_sequence(conv_type, inputs, filters, widths, strides, activation, data_format, train):
  
  conv_op = gated_conv if conv_type == 'gated_conv' else tf.layers.conv1d
  
  prev_layer = inputs
  
  for layer in range(len(filters)):
    layer_name = conv_type + '_layer_' + str(layer)
    with tf.variable_scope(layer_name):
      conv = conv_op(inputs = prev_layer,filters = filters[layer],
                     kernel_size = widths[layer], strides = strides[layer],
                     activation = activation,padding = 'same',
                     data_format = data_format, name = 'conv')
      
      prev_layer = conv
      
      tf.summary.histogram(layer_name, prev_layer)
      
  return prev_layer


def get_batch_seqs_len(inputs, data_format):
  
  len_index = -1 if data_format == 'channel_first' else 1
  
  unpack_batch = tf.unstack(inputs, axis = 0)
  
  seqs_len = [x.get_shape().as_list()[len_index] for x in unpack_batch]
  
  sequences_len = tf.cast(seqs_len, tf.int32)
  
  return sequences_len


def dense_to_sparse(dense_tensor):
  
  idx = tf.where(tf.not_equal(dense_tensor, 0))
  values = tf.gather_nd(dense_tensor, idx)
  shape = tf.shape(dense_tensor,out_type=tf.int32)
  sparse_tensor = tf.SparseTensor(idx, values , shape)
  
  return sparse_tensor
    


def teacher_model_function(features, labels, mode, params):

  print(features['shape'].get_shape())

  audio_features = features['audio']
    
  if params.get('data_format') == "channels_last":
    
    audio_features = tf.transpose(audio_features, [0, 2, 1])
    
      
  with tf.variable_scope("model"):
    
    pre_out = convolutional_sequence(inputs = audio_features, conv_type = params.get('conv_type'),
                            filters = params.get('teacher').get('filters'),
                            widths = params.get('teacher').get('widths'),
                            strides = params.get('teacher').get('strides'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            train = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.conv1d(inputs = pre_out, filters = params.get('vocab_size'), kernel_size = 1,
                                  strides= 1, activation=None,
                                  padding="same", data_format=params.get('data_format'),name="logits")
    
    
  with tf.name_scope('loss'):
    batches_ctc_loss = tf.nn.ctc_loss(labels = tf.contrib.layers.dense_to_sparse(labels),
                                      inputs =  logits, sequence_length = 100)
    loss = tf.reduce_mean(batches_ctc_loss)
    tf.summary.scalar('ctc_loss',loss)
    
  if mode == tf.estimator.ModeKeys.TRAIN:
      
    with tf.variable_scope("optimizer"):
      train_step = tf.train.AdamOptimizer().minimize(loss)
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_step) 
      
      
  elif mode == tf.estimator.ModeKeys.PREDICT:
    
    sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)
    
    dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
                                       sparse_decoded.dense_shape,
                                       sparse_decoded.values)
    
    predictions = {'decoding' : dense_decoded}
        
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  
  elif mode == tf.estimator.ModeKeys.EVAL:
    
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), labels))
    
    eval_metric_ops = {"ler": ler}
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
    
            
    
    
    
  
    
    
    
  
  
