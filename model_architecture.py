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



def teacher_model_function(features, labels, mode, params):

  audio_features = features['audio']
  
  seqs_len = features['seq_len']
  
  sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
      
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
    
    
        # get logits in time major : [max_time, batch_size, num_classes]
    if params.get('data_format') == 'channels_first':
      logits = tf.transpose(logits, (2,0,1))
      
    elif params.get('data_format') == 'channels_last':
      logits = tf.transpose(logits, (1,0,2))
      
    
  with tf.name_scope('predictions'):
    
    sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)

  if mode == tf.estimator.ModeKeys.PREDICT:
    
    print("I am here in Predict")
    
    predictions = {'decoding' : sparse_decoded, 'log_prob' : log_prob}
      
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  
  with tf.name_scope('loss'):
    
    batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                      inputs =  logits, 
                                      sequence_length = seqs_len)
    loss = tf.reduce_mean(batches_ctc_loss)
    tf.summary.scalar('ctc_loss',loss)

    
  if mode == tf.estimator.ModeKeys.TRAIN:
    
    print("I am here in Train")
    
      
    with tf.variable_scope("optimizer"):
      train_step = tf.train.AdamOptimizer().minimize(loss,
                                                      global_step=tf.train.get_global_step())
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_step) 
  
  
  assert mode == tf.estimator.ModeKeys.EVAL, "Wrong mode"

  print("Evaluating")

   
  
  ler = tf.edit_distance(tf.cast(sparse_decoded[0], tf.int32), sparse_labels)
  mean_ler, op = tf.metrics.mean(ler)
  
  metrics = {"ler": (mean_ler, op)}
  
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
  
    
            
    
    
    
  
    
    
    
  
  
