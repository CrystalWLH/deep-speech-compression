#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:12:41 2018

@author: Samuele Garda
"""

import argparse
from configparser import ConfigParser
import json
import tensorflow as tf
from model_input import teacher_input_func 
from model_architecture import teacher_model_function
from utils.transcription_utils import get_char_encoding


def parse_arguments():
  """
  Read configuration file.
  """
  
  parser = argparse.ArgumentParser(description='Run experiments for Deep ASR model compression through Teacher-Student training')
  parser.add_argument('-c', '--config', required=True, help='Architecture configuration file')
  parser.add_argument('-m', '--mode', required=True, choices = ('train','eval','predict'), 
                      help='Mode for experiment. One of :(`train`,`eval`,`predict`) ')
  args = parser.parse_args()
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read(args.config)
  return args,configuration


def config2params(configuration):
  
  map_act = {'relu' : tf.nn.relu, 'elu' : tf.nn.elu}
  
  env_params = {}
  env_params['save_models'] = configuration['GENERAL'].get('save_models','models')
  env_params['model_type'] = configuration['GENERAL'].get('model_type','teacher')
  env_params['steps'] = configuration['TRAIN'].getint('steps', 10)
  
  params = {}
  
  model_type = 'TEACHER' if env_params.get('model_type') == 'teacher' else 'STUDENT'
    
  params['filters'] = json.loads(configuration[model_type].get('filters', [250,250]))
  params['widths'] = json.loads(configuration[model_type].get('widths', [7,7]))
  params['strides'] = json.loads(configuration[model_type].get('strides', [1,1]))
  
  params['data_format'] = configuration['GENERAL'].get('data_format','channels_last')
  params['input_type'] = configuration['GENERAL'].get('input_type','ampl')
  params['conv_type'] = configuration['GENERAL'].get('conv_type','conv')
  params['vocab_size'] = len(get_char_encoding(configuration['GENERAL'].get('vocab_path','./test/vocab.pkl')))
  params['data_path'] = configuration['GENERAL'].get('data_path','./test')
  
  params['activation'] = map_act.get(configuration['TRAIN'].get('activation','relu'), 'relu')
  params['batch_size'] = configuration['TRAIN'].getint('batch_size', 512)
  params['bn'] = configuration['TRAIN'].getboolean('bn', False)
  params['temperature'] = configuration['TRAIN'].getint('temperature', 3)
  
  return env_params,params

if __name__ == '__main__':
  
  args,config_file = parse_arguments()
  
  env_params,params = config2params(config_file)
  
  config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=1, save_checkpoints_steps=2)
  
  estimator = tf.estimator.Estimator(model_fn=teacher_model_function, params=params,
                                     model_dir= env_params.get('save_models'),config=config)
    
  def input_fn():
    return teacher_input_func(tfrecord_path = './test/librispeech_tfrecords.dev',
                                                            shuffle = 10, split = 'predict', batch_size = 4 )
  

  if args.mode == "train":
    estimator.train(input_fn= input_fn,
                    steps= env_params.get('steps'))
    
  elif args.mode == "eval":
    res = estimator.evaluate(input_fn=input_fn)
    print("\n")
    print("ler : {}".format(res))

  elif args.mode == "predict":
    predictions = estimator.predict(input_fn=input_fn, yield_single_examples = False)
    for p in predictions:
      print(p['decoding'])
          

  
  


  
  
  
  
  
  
  
  

