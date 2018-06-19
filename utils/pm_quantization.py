#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:59:02 2018

@author: Samuele Garda
"""

import os
import argparse
import tensorflow as tf
from utils.quantization import quantize_uniform


def parse_args():
  """
  Parse arguments for script.
  """
  parser = argparse.ArgumentParser(description='Create checkpoint for Post Mortem Quantized Model')
  parser.add_argument('--orig', required=True, type = str, help='Path to model checkpoint to be quantized')
  parser.add_argument('--pm', required = True, type = str, help='Where to store quantized model ')
  parser.add_argument('--name', default = 'quant', type = str, help='Name for checkpoint')
  parser.add_argument('--num-bits', default = 8, type = int, help ="Number of bits for quantization. Default : 8")
  parser.add_argument('--bucket', default = 256, type = int, help ="Bucket size for bucketing in quantization. Pass 0 for no bucketing. Default : 256")
  parser.add_argument('--stochastic', action = "store_true", help ="Use stochastic quantization")
                      
  return parser.parse_args()


if __name__ == "__main__":
  
  args = parse_args()
  
  orig_model = args.orig
  pm = args.pm
  name = args.name
  
  bkt_size = args.bucket
  s = 2 ** args.num_bits 
  stoch = args.stochastic
  
  if not os.path.exists(pm):
    os.makedirs(pm)
    
  
  tf.reset_default_graph()
  
  with tf.Session() as sess:

    print('Load checkpoint of original model')
    saver = tf.train.import_meta_graph(orig_model + '.meta')
    print('Restore variables')
    saver.restore(sess, orig_model)
    
    pm_saver = tf.train.Saver()
    
    # get your variable
    variables = tf.trainable_variables()
    print("Variables to quantize : {}".format(len(variables)))
    
    for idx,var in enumerate(variables):

      var_up = var.assign(quantize_uniform(var,s,bkt_size,stoch))
      
      print("Quantized var : {}".format(var))
      
      sess.run(var_up)
          
    pm_saver.save(sess, os.path.join(pm,name))
    print("Successfully created checkpoint with quantized variables")
    
    