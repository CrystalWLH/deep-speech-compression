#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:50:11 2018

@author: Samuele Garda
"""

import os
import logging
import argparse
import numpy as np
import tensorflow as tf
from input_funcs import teacher_input_func
from main import config2params
from utils.net import _conv_seq,_compute_logits
from utils.data2tfrecord import create_tfrecords_folder,_float_feature,_int64_feature

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def parse_args():
  """
  Parse arguments for script.
  """
  parser = argparse.ArgumentParser(description='Create tfrecord files for teacher logits')
  parser.add_argument('-p', '--path', default = './tfrecords_data', type = str, help='Folder where to store teacher guidance (logits,hints)')
  parser.add_argument('-t', '--teacher', required=True, type = str, help='Path to teacher checkpoint dir')
  parser.add_argument('-g', '--guidance', required=True, type = str, choices = ('logits','hints'), help='Which type of guidance to save. Logits for DS or Hint for FitNet')
  parser.add_argument('--up-to', default = 2, type = int,help='Which layer is the hint layer. ONLY FOR FitNet. ')
  parser.add_argument('--conf', required=True, type = str, help='Path to teacher model configuration')

  return parser.parse_args()


def tfrecord_write_ex_guidance(writer,guide,shape):
  """
  Write example to TFRecordWriter.
  
  :param:
    writer (tf.python_io.TFRecordWriter) : file where examples will be stored
    logits (np.ndarry) : audio
    shape (list) : shape of logits
  """
  
  example = tf.train.Example( features=tf.train.Features(
      feature={'guide': _float_feature(guide),
               'shape' : _int64_feature(shape)}))
  
  writer.write(example.SerializeToString())
  

if __name__  == "__main__":
  
  args = parse_args()
  
  guid = args.guidance
  hint = args.up_to
  checkpoint = tf.train.get_checkpoint_state(args.teacher)
  
  print(checkpoint.model_checkpoint_path)

  # NET CONFIGURATION
  env_params, params= config2params(args.conf)
  
  logger.info("Define input")
  
  # GET INPUT
  features,labels = teacher_input_func(tfrecord_path = env_params.get('train_data'),
                              input_channels = env_params.get('input_channels'),
                              mode = 'predict', 
                              epochs = 1,
                              batch_size = 1 )
  
  logger.info("Define network architecture")
  
  if params.get('data_format') == "channels_last":
    
    features = tf.transpose(features, (0, 2, 1))
    
  with tf.variable_scope("model"):  
  # DEFINE NETWORK
    guidance = _conv_seq(conv_type = params.get('conv_type'),
                           inputs = features,
                           filters = params.get('filters') if guid == 'logits' else params.get('filters')[:hint],
                           widths = params.get('widths') if guid == 'logits' else params.get('widths')[:hint],
                           strides = params.get('strides') if guid == 'logits' else params.get('strides')[:hint],
                           dropouts = params.get('dropouts') if guid == 'logits' else params.get('dropouts')[:hint],
                           activation = params.get('activation'),
                           data_format = params.get('data_format'),
                           batchnorm = params.get('batchnorm'),
                           train = False,
                           hist = False
                           )
    
    
    if guid == 'logits':
      
      guidance = _compute_logits(prev_layer = guidance,
                                 vocab_size = params.get('vocab_size'),
                                 data_format = params.get('data_format'))
  
  
  
  base_name = os.path.basename(env_params.get('save_model'))
  
  out_path = create_tfrecords_folder(args.path)
  
  out_file = str(out_path.joinpath(base_name)) + '.{}'.format(guid)
  
  writer = tf.python_io.TFRecordWriter(out_file)
  
  
  logger.info("Start writing `{}` in `{}`\n".format(guid,out_file))
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    count = 1
    
    while True:
      
      try:
        guide = sess.run(guidance)
        guide = np.squeeze(guide) # channels last
        guide_shape = list(guide.shape)
        guide = guide.flatten()
        tfrecord_write_ex_guidance(writer = writer, guide = guide, shape = guide_shape)
        
        count += 1
        
        if count == 2:
          
          logger.info("Guide size is : {}".format(guide_shape[1]))
        
        if (count%10000) == 0:
          logger.info("Successfully wrote {} {} to {}".format(count,guid,out_file))
      
      except tf.errors.OutOfRangeError:
        
        logger.info("Completed iterating through training data set")
        break
        
  writer.close()
  