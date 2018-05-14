#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:50:11 2018

@author: Samuele Garda
"""

import tensorflow as tf
from configparser import ConfigParser

from model_input import load_teacher_logits,teacher_input_func,load_dataset

from model_architecture import teacher_model_function
from model_main import config2params


if __name__ == "__main__":
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read('./configs/wav2letter_v1.config')
  
  teacher_env_config,teacher_params = config2params(configuration)
  
  def input_fn():
    return teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                                             split = 'train', batch_size = 2 )
  
  dataset_logits = load_teacher_logits(input_fn = input_fn,
                                       teacher_model_function = teacher_model_function,
                                       params_teacher = teacher_params,
                                       model_dir = './models/w2l_v1')
  
  
 
  dataset_standard = load_dataset('./test/librispeech_tfrecords.dev')
  
  
  merged = tf.data.Dataset.zip((dataset_standard,dataset_logits))
  
  merged = merged.padded_batch(1, padded_shapes= (([257,-1], [-1]), [-1,29] ),
                                             padding_values = ( ( 0. , -1), 1. ))
#  
#  
  next_element = merged.make_one_shot_iterator().get_next()
  
  with tf.Session() as sess:
    for i in range(3):
      try:
        features= sess.run(next_element)
        print(features)
      except tf.errors.OutOfRangeError:
        print("End of dataset")
    
  
  