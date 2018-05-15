#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:12:41 2018

@author: Samuele Garda
"""
#TODO
# CTC CRASHES ON BATCH_SIZE = 1

import argparse
import logging
from configparser import ConfigParser
import json
import tensorflow as tf
from model_input import teacher_input_func, student_input_func  
from model_architecture import teacher_model_function, student_model_function
from utils.transcription_utils import load_pickle,decoder_dict,decode_sequence

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_arguments():
  """
  Read configuration file.
  """
  
  parser = argparse.ArgumentParser(description='Run experiments for Deep ASR model compression through Teacher-Student training')
  parser.add_argument('-c', '--config', required=True, help='Experiment configuration file')
  parser.add_argument('-m', '--mode', required=True, choices = ('train','eval','predict'), 
                      help='Mode for experiment. One of :(`train`,`eval`,`predict`) ')
  args = parser.parse_args()
  
  return args


def config2params(config):
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read(config)

  map_act = {'relu' : tf.nn.relu, 'elu' : tf.nn.elu}
  
  env_params = {}
  params = {}
  
  env_params['model_type'] = configuration['GENERAL'].get('model_type','teacher')
  env_params['save_model'] = configuration['GENERAL'].get('save_model','models')
  env_params['steps'] = configuration['TRAIN'].getint('steps', 10)
  env_params['char2idx'] = load_pickle(configuration['GENERAL'].get('vocab_path','./test/vocab.pkl'))
  env_params['input_channels'] = configuration['GENERAL'].getint('input_channels', None)
  env_params['batch_size'] = configuration['TRAIN'].getint('batch_size', 512)
  
  if not env_params['input_channels']:
    logger.warning("Number of input channels is not specified! Please provide this field")
  
  model_type = 'TEACHER' if env_params.get('model_type') == 'teacher' else 'STUDENT'
  
  if model_type == 'STUDENT':
  
    env_params['teacher_config'] = configuration['GENERAL'].get('teacher_config')
    env_params['teacher_dir'] = configuration['GENERAL'].get('teacher_dir')
    params['temperature'] = configuration['TRAIN'].getint('temperature', 3)
  
    
  params['filters'] = json.loads(configuration[model_type].get('filters', [250,250]))
  params['widths'] = json.loads(configuration[model_type].get('widths', [7,7]))
  params['strides'] = json.loads(configuration[model_type].get('strides', [1,1]))
  
  
  params['vocab_size'] = len(env_params.get('char2idx'))
  params['data_format'] = configuration['GENERAL'].get('data_format','channels_last')
  params['conv_type'] = configuration['GENERAL'].get('conv_type','conv')
  params['data_path'] = configuration['GENERAL'].get('data_path','./test')
  params['activation'] = map_act.get(configuration['TRAIN'].get('activation','relu'), 'relu')
  params['bn'] = configuration['TRAIN'].getboolean('bn', False)
  
  
  return env_params,params

if __name__ == '__main__':
  
  args = parse_arguments()
  
  env_params,params = config2params(args.config)
  
  config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=1, save_checkpoints_steps=2)
  
  
  if env_params.get('model_type') == 'teacher':
  
    estimator = tf.estimator.Estimator(model_fn=teacher_model_function, params=params,
                                     model_dir= env_params.get('save_model'),config=config)
    
    if args.mode == "train":
      
      def input_fn():
        return teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                  input_channels = env_params.get('input_channels'),
                                  mode = 'train',
                                  batch_size = 2 ) #env_params.get('batch_size')
                                  

      estimator.train(input_fn= input_fn,steps= env_params.get('steps'))
      
    elif args.mode == "eval":
      
      def input_fn():
        return teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                    input_channels = env_params.get('input_channels'),
                                    mode = 'eval',
                                    batch_size = 5)
      
      
      res = estimator.evaluate(input_fn=input_fn)
      print("evaluation : {}".format(res))
  
    elif args.mode == "predict":
      
      def input_fn():
        return teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                  input_channels = env_params.get('input_channels'),
                                  mode = 'predict', 
                                  batch_size = 2 )
      
      idx2char = decoder_dict(env_params.get('char2idx'))
      for batch_pred in estimator.predict(input_fn=input_fn, yield_single_examples = False):
        for pred in batch_pred['decoding']:
          print(decode_sequence(pred,idx2char))
          
          
  elif env_params.get('model_type') == 'student':
    
    env_teacher,params_teacher = config2params(env_params.get('teacher_config'))
    
    
    estimator = tf.estimator.Estimator(model_fn=student_model_function, params=params,
                                       model_dir= env_params.get('save_model'),config=config)
    
    if args.mode == 'train':
      
      def input_fn():
        return student_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                  vocab_size = len(env_params.get('char2idx')),
                                  input_channels = env_params.get('input_channels'), 
                                  mode = 'train',
                                  batch_size = 5, #env_params.get('batch_size')
                                  teacher_model_function = teacher_model_function,
                                  params_teacher = params_teacher,
                                  model_dir = env_params.get('teacher_dir'))
        
      estimator.train(input_fn= input_fn,steps= env_params.get('steps'))
        
        
        
        
        
        
      
      

  
  


  
  
  
  
  
  
  
  

