#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:12:41 2018

@author: Samuele Garda
"""

import argparse
import logging
from configparser import ConfigParser
import json
import tensorflow as tf
from model_input import teacher_input_func, student_input_func  
from model_architecture import teacher_model_function, student_model_function
from utils.create_tfrecords import load_pickle
from utils.transcription_utils import decoder_dict,decode_sequence

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_arguments():
  """
  Read configuration file.
  """
  
  parser = argparse.ArgumentParser(description='Run experiments for Deep ASR model compression through Teacher-Student training')
  parser.add_argument('-m', '--mode', required=True, choices = ('train','eval','predict'), 
                      help='Mode for experiment. One of :(`train`,`eval`,`predict`) ')
  parser.add_argument('--conf', required=True, help='Experiment configuration file')
  args = parser.parse_args()
  
  return args


def complete_name(env_params,params):
  """
  Create name excplicting network hyperparameters.
  
  :param:
    env_params (dict) : parameters for experiment run
    params (dict) : network parameters
    
  :return:
    model_name (str) : explicit model name
  """
  
  act_map = {tf.nn.relu : 'relu', tf.nn.elu : 'elu'}
  
  if env_params.get('model_type') == 'teacher':
    
    model_name = "{}_bn{}_bs{}_{}_c{}_{}{}_do{}".format(env_params.get('save_model'),int(params.get('bn')),env_params.get('batch_size'),
                  act_map[params.get('activation')],params.get('clipping'),params.get('conv_type'),
                  len(params.get('filters')),params.get('dropouts')[-1])
  
  if env_params.get('model_type') == 'student':
  
    model_name = "{}_bn{}_bs{}_{}_c{}_{}{}_do{}_t{}".format(env_params.get('save_model'),int(params.get('bn')),env_params.get('batch_size'),
                  act_map[params.get('activation')],params.get('clipping'),params.get('conv_type'),
                  len(params.get('filters')),params.get('dropouts')[-1],
                  params.get('temperature'))
    
  
  return model_name
  

def config2params(config):
  """
  Create configuration varibales both for experiment run and model.
  
  :params:
    config (str) : path to configuration file
  :return:
    env_params (dict) : parameters for experiment run
    params (dict) : network parameters
    
  """
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read(config)

  map_act = {'relu' : tf.nn.relu, 'elu' : tf.nn.elu}
  
  env_params = {}
  params = {}
  
  env_params['model_type'] = configuration['GENERAL'].get('model_type','teacher')
  params['model_type'] = env_params['model_type']
  env_params['save_model'] = configuration['GENERAL'].get('save_model','models')
  env_params['epochs'] = configuration['TRAIN'].getint('epochs', 50)
  env_params['input_channels'] = configuration['GENERAL'].getint('input_channels', None)
  env_params['batch_size'] = configuration['TRAIN'].getint('batch_size', 512)
  env_params['train_data'] = configuration['FILES'].get('train_data','./tfrecords_data/tfrecords_mfccs.train')
  env_params['eval_data'] = configuration['FILES'].get('train_data','./tfrecords_data/tfrecords_mfccs.dev_clean')
  env_params['test_data'] = configuration['FILES'].get('train_data','./tfrecords_data/tfrecords_mfccs.test_clean')
  env_params['char2idx'] = load_pickle(configuration['FILES'].get('vocab_path','./test/vocab.pkl'))
  
  if not env_params['input_channels']:
    logger.warning("Number of input channels is not specified! Please provide this field")
  
  model_type = 'TEACHER' if env_params.get('model_type') == 'teacher' else 'STUDENT'
  
  if model_type == 'STUDENT':
  
    env_params['teacher_logits'] = configuration['FILES'].get('teacher_logits')
    params['temperature'] = configuration['TRAIN'].getint('temperature', 3)
    params['alpha'] = configuration['TRAIN'].getfloat('alpha', 0.3)
    
  params['filters'] = json.loads(configuration[model_type].get('filters', [250,250]))
  params['widths'] = json.loads(configuration[model_type].get('widths', [7,7]))
  params['strides'] = json.loads(configuration[model_type].get('strides', [1,1]))
  params['dropouts'] = json.loads(configuration[model_type].get('dropouts', [0,0]))
  
  lengths = list(map(len, [params['filters'], params['widths'], params['strides'],params['dropouts']  ] ))
  
  if len(set(lengths)) > 1:
    raise ValueError("Plaese check the net definition in {} section! All the lists must have same length! \n \
                   Found : `filters` : {},`widths` : {},`strides` : {},`dropouts` : {}".format(model_type,*lengths))
  
  params['adam_lr'] = configuration['TRAIN'].getfloat('adam_lr',1e-4)
  params['adam_eps'] = configuration['TRAIN'].getfloat('adam_eps',1e-8)
  params['vocab_size'] = len(env_params.get('char2idx'))
  params['data_format'] = configuration['GENERAL'].get('data_format','channels_last')
  params['conv_type'] = configuration['GENERAL'].get('conv_type','conv')
  params['activation'] = map_act.get(configuration['TRAIN'].get('activation','relu'), 'relu')
  params['bn'] = configuration['TRAIN'].getboolean('bn', False)
  params['clipping'] = configuration['TRAIN'].getint('clipping', 0)
  
  
  return env_params,params

if __name__ == '__main__':
  
  args = parse_arguments()
  
  env_params,params = config2params(args.conf)
    
  config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=1, save_checkpoints_steps=20)
  
  logging_hook = tf.train.LoggingTensorHook({"mean_ler": "mean_ler"}, every_n_iter=1000)
  
  
  if env_params.get('model_type') == 'teacher':
  
    estimator = tf.estimator.Estimator(model_fn=teacher_model_function, params=params,
                                     model_dir= complete_name(env_params,params),
                                     config=config)
    
    if args.mode == "train":
      
      def input_fn():
        return teacher_input_func(tfrecord_path = env_params.get('train_data'),
                                  input_channels = env_params.get('input_channels'),
                                  mode = 'train',
                                  epochs = env_params.get('epochs'),
                                  batch_size = env_params.get('batch_size'))
                                  

      estimator.train(input_fn= input_fn,steps= env_params.get('steps'))
      
    elif args.mode == "eval":
      
      def input_fn():
        return teacher_input_func(tfrecord_path = env_params.get('eval_data'),
                                    input_channels = env_params.get('input_channels'),
                                    mode = 'eval',
                                    epochs = 1,
                                    batch_size = env_params.get('batch_size'))
      
      
      res = estimator.evaluate(input_fn=input_fn)
      print("evaluation : {}".format(res))
  
    elif args.mode == "predict":
      
      def input_fn():
        return teacher_input_func(tfrecord_path = env_params.get('test_data'),
                                  input_channels = env_params.get('input_channels'),
                                  mode = 'predict', 
                                  epochs = 1,
                                  batch_size = 1 )
      
      idx2char = decoder_dict(env_params.get('char2idx'))
      for batch_pred in estimator.predict(input_fn=input_fn, yield_single_examples = False):
        for p in batch_pred['decoding']:
          print(decode_sequence(p,idx2char))
          
          
  elif env_params.get('model_type') == 'student':
    
    estimator = tf.estimator.Estimator(model_fn=student_model_function, params=params,
                                       model_dir= complete_name(env_params,params),
                                       config=config)
    
    
    if args.mode == 'train':
      
      def input_fn():
        return student_input_func(tfrecord_path = env_params.get('train_data'),
                                  tfrecord_logits = env_params.get('teacher_logits'),
                                  vocab_size = len(env_params.get('char2idx')),
                                  input_channels = env_params.get('input_channels'), 
                                  mode = 'train',
                                  epochs = env_params.get('epochs'),
                                  batch_size = env_params.get('batch_size')
                                  )
        
      estimator.train(input_fn= input_fn)
        
        
        
        
        
        
      
      

  
  


  
  
  
  
  
  
  
  

