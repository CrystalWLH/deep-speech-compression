#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:40:51 2018

@author: Samuele Garda
"""
import tensorflow as tf
from model_architecture import convolutional_sequence
from model_input import student_input_func
from model_main import config2params

env_params,params = config2params('./configs/student.config')

features, labels = student_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                  tfrecord_logits = './test/w2l_v1.logits',
                                  vocab_size = 29,
                                  input_channels = 257, 
                                  mode = 'train',
                                  batch_size = 2 #env_params.get('batch_size')
                                  )


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


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for _ in range(1000):
    try:
      test_teacher_logits,student_logits = sess.run([teacher_logits_shape,student_logits_shape])
      print(test_teacher_logits,student_logits)
      assert test_teacher_logits == student_logits, "Error - Teacher logit shape {} - Student logit shape {}".format(test_teacher_logits,student_logits)
    except tf.errors.OutOfRangeError:
      print("End data")
      
    