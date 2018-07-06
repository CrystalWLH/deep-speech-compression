#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:59:02 2018

@author: Samuele Garda
"""

import os
import logging
import argparse
import tensorflow as tf
from utils.quantization import quantize_uniform

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_args():
  """
  Parse arguments for script.
  """
  parser = argparse.ArgumentParser(description='Create checkpoint for Post Mortem Quantized Model')
  parser.add_argument('--orig', required=True, type = str, help='Path to model directory')
  parser.add_argument('--pm', required = True, type = str, help='Where to store quantized model ')
  parser.add_argument('--num-bits', default = 8, type = int, help ="Number of bits for quantization. Default : 8")
  parser.add_argument('--bucket', default = 256, type = int, help ="Bucket size for bucketing in quantization. Pass 0 for no bucketing. Default : 256")
  parser.add_argument('--stochastic', action = "store_true", help ="Use stochastic quantization")
  parser.add_argument('--quant-last', action = "store_true", help ="Quantize last layer. ONLY FOR MODEL TRAINED WITH QUANTIZED DISTILLATION")
                      
  return parser.parse_args()


if __name__ == "__main__":
  
  args = parse_args()
  
  checkpoint = tf.train.get_checkpoint_state(args.orig)
  pm = args.pm
  
  bkt_size = args.bucket
  s = 2 ** args.num_bits 
  stoch = args.stochastic
  
  
  if not os.path.exists(pm):
    os.makedirs(pm)
    
  
  tf.reset_default_graph()
  
  with tf.Session() as sess:

    logger.info('Load checkpoint of original model')
    saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
    logger.info('Restore variables')
    saver.restore(sess, checkpoint.model_checkpoint_path)
    
    pm_saver = tf.train.Saver()
    
    # get your variable
    variables = tf.trainable_variables()
    logger.info("Variables to quantize : {}".format(len(variables)))
    print(variables)    
    for idx,var in enumerate(variables):
      
      var_up = var.assign(quantize_uniform(var,s,bkt_size,stoch))
        
      sess.run(var_up)
    
    folder =  [part for part in checkpoint.model_checkpoint_path.split(os.path.sep) if 'conv' in part][0]
    ckpt_name =  os.path.basename(os.path.normpath(checkpoint.model_checkpoint_path))
    full_path = os.path.join(pm,folder,ckpt_name)
    
    pm_saver.save(sess,full_path )
    logger.info("Successfully created checkpoint with quantized variables at `{}`".format(full_path))
    
    