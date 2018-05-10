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
from model_input import model_input_func_tfr 
from model_architecture import teacher_model_function
from utils.transcription_utils import get_char_encoding


def parse_arguments():
  """
  Read configuration file.
  """
  
  parser = argparse.ArgumentParser(description='Run experiments for Deep ASR model compression through Teacher-Student training')
  parser.add_argument('-c', '--config', required=True, help='Architecture configuration file')
  args = parser.parse_args()
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read(args.config)
  return configuration


def config2params(configuration):
  
  map_act = {'relu' : tf.nn.relu, 'elu' : tf.nn.elu}
  
  params = {}
  params['teacher'] = {}
  params['student'] = {}
  
  
  params['data_format'] = configuration['GENERAL'].get('data_format','channels_last')
  params['save_models'] = configuration['GENERAL'].get('save_models','models')
  params['input_type'] = configuration['GENERAL'].get('input_type','ampl')
  params['conv_type'] = configuration['GENERAL'].get('input_type','conv')
  params['vocab_size'] = len(get_char_encoding(configuration['GENERAL'].get('vocab_path','./test/vocab.pkl')))
  params['data_path'] = configuration['GENERAL'].get('data_path','./test')
  params['mode'] =  configuration['GENERAL'].get('mode','train')
  
  
  params['train'] = configuration['TRAIN'].get('train','teacher')
  params['activation'] = map_act.get(configuration['TRAIN'].get('activation','relu'))
  
  params['batch_size'] = configuration['TRAIN'].getint('batch_size', 512)
  params['steps'] = configuration['TRAIN'].getint('steps', 10)
  params['bn'] = configuration['TRAIN'].getboolean('bn', False)
  params['temperature'] = configuration['TRAIN'].getint('temperature', 3)
  
  params['teacher']['filters'] = json.loads(configuration['TEACHER'].get('filters', [250,250]))
  params['teacher']['widths'] = json.loads(configuration['TEACHER'].get('widths', [7,7]))
  params['teacher']['strides'] = json.loads(configuration['TEACHER'].get('strides', [1,1]))
  
  params['student']['filters'] = json.loads(configuration['STUDENT'].get('filters', [250,250]))
  params['student']['widths'] = json.loads(configuration['STUDENT'].get('widths', [7,7]))
  params['student']['strides'] = json.loads(configuration['STUDENT'].get('strides', [1,1]))
  
  return params

if __name__ == '__main__':
  
  args = parse_arguments()
  
  params = config2params(args)
  
  config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=1, save_checkpoints_steps=2)
  
  estimator = tf.estimator.Estimator(model_fn=teacher_model_function, params=params,
                                     model_dir= params.get('save_models'),config=config)
  
  logging_hook = tf.train.LoggingTensorHook({"ler": "ler"}, every_n_iter=2)
    
  if params.get('mode') == 'train':
    estimator.train(input_fn= lambda : model_input_func_tfr(tfrecord_path = './test/librispeech_tfrecords.dev',
                                                            shuffle = 10,split = 'dev', batch_size = 2 ),
                    steps= params.get('steps'),
                    hooks=[logging_hook])
    
#  elif params.get('mode') == 'predict':
#    pred = estimator.predict(input_fn=input_fn)
#    
#    print(pred)
#    
#  elif params.get('mode') == "eval":
#    res = estimator.evaluate(input_fn=input_fn)
#    print("\n")
#    print("ler : {}".format(res))
#
#  
#  
#
#
#  
  
  
  
  
  
  
  

