#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:18:06 2018

@author: Samuele Garda
"""

import tensorflow as tf
from tfrecords_utils import load_tfrecord_dataset

if __name__ == "__main__":
  
  dataset = load_tfrecord_dataset('./test/librispeech_tfrecords.dev', 
                                   split = 'dev', batch_size = 1)
  
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()  
  
  
  with tf.Session() as sess:
    for i in range(11):
      try:
        value = sess.run(next_element)
        print(value)
        print('\n\n')
      except tf.errors.OutOfRangeError:
        print("End of dataset")

    
    