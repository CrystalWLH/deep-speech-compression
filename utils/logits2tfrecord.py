#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:50:11 2018

@author: Samuele Garda
"""

import logging
import argparse
import numpy as np
import tensorflow as tf
from input_funcs import teacher_input_func
from models import teacher_model_function
from main import config2params,complete_name
from utils.data2tfrecord import create_tfrecords_folder,_float_feature,_int64_feature

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def parse_args():
  """
  Parse arguments for script.
  """
  parser = argparse.ArgumentParser(description='Create tfrecord files for teacher logits')
  parser.add_argument('-p', '--path', default = './tfrecords_data', type = str, help='Folder where to store logits')
  parser.add_argument('-n', '--model-name', required=True, type = str, help='Identifier for which model produced the logits')
  parser.add_argument('--conf', required=True, type = str, help='Path to teacher model configuration')

  return parser.parse_args()



def tfrecord_write_ex_logits(writer,logits,shape):
  """
  Write example to TFRecordWriter.
  
  :param:
    writer (tf.python_io.TFRecordWriter) : file where examples will be stored
    logits (np.ndarry) : audio
    shape (list) : shape of logits
  """
  
  example = tf.train.Example( features=tf.train.Features(
      feature={'logits': _float_feature(logits),
               'shape' : _int64_feature(shape)}))
  
  writer.write(example.SerializeToString())
  

if __name__  == "__main__":
  
  args = parse_args()

  env_teacher, params_teacher = config2params(args.conf)
  
  
  def input_fn():
    return teacher_input_func(tfrecord_path = env_teacher.get('train_data'),
                              input_channels = env_teacher.get('input_channels'),
                              mode = 'predict', 
                              epochs = 1,
                              batch_size = 1 )
  
  estimator = tf.estimator.Estimator(model_fn=teacher_model_function, params=params_teacher,
                                       model_dir= complete_name(env_teacher,params_teacher))
  
  
  out_path = create_tfrecords_folder(args.path)
  
  out_file = str(out_path.joinpath(args.model_name)) + '.logits'
  
  writer = tf.python_io.TFRecordWriter(out_file)
  
  for idx,batch_pred in enumerate(estimator.predict(input_fn=input_fn, yield_single_examples = False), start = 1):
    
    logits = np.squeeze(batch_pred['logits'])
    
    logits_shape = list(logits.shape)
        
    logits = logits.flatten()
    
    tfrecord_write_ex_logits(writer = writer, logits = logits, shape = logits_shape)
    
    if (idx%10000) == 0:
      logger.info("Successfully wrote {} logits to {}".format(idx,out_file))
       
  writer.close()
