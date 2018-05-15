#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:50:11 2018

@author: Samuele Garda
"""

import tensorflow as tf
import numpy as np

from model_input import load_teacher_logits, teacher_input_func,student_input_func
from model_architecture import teacher_model_function
from model_main import config2params



env_teacher, params_teacher = config2params('./configs/wav2letter_v1.config')


features,labels = student_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                  vocab_size = 29,
                                  input_channels = 257, 
                                  mode = 'train',
                                  batch_size = 1, #env_params.get('batch_size')
                                  teacher_model_function = teacher_model_function,
                                  params_teacher = params_teacher,
                                  model_dir = './models/w2l_v1')


teacher_f, teacher_l = teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                  input_channels = 257,
                                  mode = 'predict', 
                                  batch_size = 1 )

#dataset_logits = load_teacher_logits(tfrecord_path_train = './test/librispeech_tfrecords.dev',
#                                  input_channels = 257, 
#                                  teacher_model_function = teacher_model_function,
#                                  params_teacher = params_teacher,
#                                  model_dir = './models/w2l_v1')
#
#
#dataset_logits = dataset_logits.batch(1)
#logits = dataset_logits.make_one_shot_iterator().get_next()
#
#  
#estimator = tf.estimator.Estimator(model_fn=teacher_model_function, params=params_teacher,
#                                   model_dir= './models/w2l_v1')
#
#def input_fn():
#  return teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
#                                input_channels = 257,
#                                mode = 'predict', 
#                                batch_size = 1 )
#  
##features,labels =  teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
##                                input_channels = 257,
##                                mode = 'predict', 
##                                batch_size = 1 )
##  
##print(features)
#  
#
#
#ordered_shape_pred = []
#pred = estimator.predict(input_fn=input_fn, yield_single_examples = False)
#for idx,batch_pred in enumerate(pred):
#  ordered_shape_pred.append(batch_pred['logits'].shape)
#  
#  
#  
#ordered_shapes_input = []
with tf.Session() as sess:
  for i in range(100):
    try:
      st_features,t_features = sess.run([features,teacher_f])
      print(st_features['logits'].shape, t_features.shape,sep = '->')
    except tf.errors.OutOfRangeError:
      print("End of dataset")  
#      
#      
#for pred,logits in zip(ordered_shape_pred,ordered_shapes_input):
#  print(pred,logits,sep = '->')
#    
