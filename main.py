#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:50:09 2018

@author: Samuele Garda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:12:41 2018

@author: Samuele Garda
"""

import os
import argparse
import logging
from configparser import ConfigParser
import json
import tensorflow as tf
from input_funcs import teacher_input_func, student_input_func  
from models_class import TeacherModel,StudentModel,QuantStudentModel
from utils.transcription import load_chars2id_from_file

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
  
  
  # ENVIRONMENT PARAMS (e.g. : model type, number of input channels,data files,...)
  env_params['model_type'] = configuration['GENERAL'].get('model_type','teacher')
  env_params['save_model'] = configuration['GENERAL'].get('save_model','models')
  env_params['epochs'] = configuration['TRAIN'].getint('epochs', 50)
  env_params['input_channels'] = configuration['GENERAL'].getint('input_channels', None)
  env_params['batch_size'] = configuration['TRAIN'].getint('batch_size', 512)
  env_params['train_data'] = configuration['FILES'].get('train_data',os.path.join('tfrecords_data','tfrecords_mfccs.train'))
  env_params['eval_data'] = configuration['FILES'].get('eval_data',os.path.join('tfrecords_data','tfrecords_mfccs.dev-clean'))
  env_params['test_data'] = configuration['FILES'].get('test_data',os.path.join('tfrecords_data','tfrecords_mfccs.test-clean'))
  env_params['vocab_size'] =  len(load_chars2id_from_file(configuration['FILES'].get('vocab_path',os.path.join('tfrecords_data','ctc_vocab.txt'))))
  env_params['knlem_op'] = configuration['LM'].get('knlem_op',os.path.join('lm_op','libctc_decoder_with_kenlm.so'))
  
  
  if not env_params['input_channels']:
    raise ValueError("Number of input channels is not specified! Please provide!")
  
  # GENERAL PARAMETERS FOR ALL MODELS
  params['model_type'] = env_params['model_type']
  params['char2idx'] = configuration['FILES'].get('vocab_path',os.path.join('tfrecords_data','ctc_vocab.txt'))
  params['vocab_size'] = env_params['vocab_size']
  params['adam_lr'] = configuration['TRAIN'].getfloat('adam_lr',1e-4)
  params['adam_eps'] = configuration['TRAIN'].getfloat('adam_eps',1e-8)
  params['data_format'] = configuration['GENERAL'].get('data_format','channels_last')
  params['conv_type'] = configuration['GENERAL'].get('conv_type','conv')
  params['activation'] = map_act.get(configuration['TRAIN'].get('activation','relu'), 'relu')
  params['bn'] = configuration['TRAIN'].getboolean('bn', False)
  params['clipping'] = configuration['TRAIN'].getint('clipping', 0)
  
  model_type = 'TEACHER' if env_params.get('model_type') == 'teacher' else 'STUDENT'
  
  # PARAMETERS SPECIFIC TO STUDENT MODELS
  if model_type == 'STUDENT':
  
    env_params['teacher_logits'] = configuration['FILES'].get('teacher_logits')
    env_params['quantize'] = configuration['GENERAL'].getboolean('quantize',False)
    params['temperature'] = configuration['TRAIN'].getint('temperature', 3)
    params['alpha'] = configuration['TRAIN'].getfloat('alpha', 0.3)
    
    # PARAMETERS SPECIFIC TO QUANTIZED STUDENT MODELS
    if env_params.get('quantize'):
      params['num_bits'] = configuration['QUANTIZATION'].getint('num_bits', 4)
      params['bucket_size'] = configuration['QUANTIZATION'].getint('bucket_size', 256)
      params['stochastic'] = configuration['QUANTIZATION'].getboolean('stochastic', False)
      params['quant_last_layer'] = configuration['QUANTIZATION'].getboolean('quant_last_layer', False)
  
  # ARCHITECTURE      
  params['filters'] = json.loads(configuration[model_type].get('filters', [250,250]))
  params['widths'] = json.loads(configuration[model_type].get('widths', [7,7]))
  params['strides'] = json.loads(configuration[model_type].get('strides', [1,1]))
  params['dropouts'] = json.loads(configuration[model_type].get('dropouts', [0,0]))
  
  lengths = list(map(len, [params['filters'], params['widths'], params['strides'],params['dropouts']  ] ))
  
  if len(set(lengths)) > 1:
    raise ValueError("Plaese check the network definition in {} section! All the lists must have same length! \n \
                   Found : `filters` : {},`widths` : {},`strides` : {},`dropouts` : {}".format(model_type,*lengths))  
    
  
  # LANGUAGE MODEL PARAMETERS
  params['lm'] = configuration['LM'].getboolean('lm', False)
  params['beam_search'] = configuration['LM'].getboolean('beam_search', False)
  params['lm_binary'] = configuration['LM'].get('lm_binary',os.path.join('lm_data','lm.binary'))
  params['lm_trie'] = configuration['LM'].get('lm_trie',os.path.join('lm_data','trie'))
  params['lm_alphabet'] = configuration['LM'].get('lm_alphabet',os.path.join('lm_data','alphabet.txt'))
  params['lm_weight'] = configuration['LM'].getfloat('lm_weight',1.75)
  params['word_count_weight'] = configuration['LM'].getfloat('word_count_weight',1.00)
  params['valid_word_count_weight'] = configuration['LM'].getfloat('valid_word_count_weight',1.00)
  params['top_paths'] = configuration['LM'].getint('top_paths',1)
  params['beam_width'] = configuration['LM'].getint('beam_width',1024)
  
  return env_params,params

if __name__ == '__main__':
  
  args = parse_arguments()
  
  env_params,params = config2params(args.conf)
  
 
  config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=1, save_checkpoints_steps=1000)
  
  if env_params.get('model_type') == 'teacher':
    
    model = TeacherModel(custom_op = env_params.get('knlem_op'))
      
    def input_fn():
      return teacher_input_func(tfrecord_path = env_params.get('train_data'),
                                input_channels = env_params.get('input_channels'),
                                mode = args.mode,
                                epochs = env_params.get('epochs'),
                                batch_size = env_params.get('batch_size') if args.mode == 'train' else 1
                                )
      
  elif env_params.get('model_type') == 'student':
    
    if not env_params.get('quantize'):
      
      model = StudentModel(custom_op = env_params.get('knlem_op'))
    
    else:
      
      model = QuantStudentModel(custom_op = env_params.get('knlem_op')) 
      
  
    def input_fn():
      return student_input_func(tfrecord_path = env_params.get('train_data'),
                                tfrecord_logits = env_params.get('teacher_logits'),
                                vocab_size = env_params.get('vocab_size'),
                                input_channels = env_params.get('input_channels'), 
                                mode = args.mode,
                                epochs = env_params.get('epochs'),
                                batch_size =env_params.get('batch_size') if args.mode == 'train' else 1
                                )
  
  estimator = tf.estimator.Estimator(model_fn= model.model_function, params=params,
                                     model_dir= complete_name(env_params,params),
                                     config=config)

  if args.mode == 'train':
  
    estimator.train(input_fn= input_fn)
    
  elif args.mode == 'eval':
    
    estimator.evaluate(input_fn=input_fn)
    
  elif args.mode == 'predict':
    
    for idx,batch_pred in enumerate(estimator.predict(input_fn=input_fn, yield_single_examples = False)):
      if idx == 0:
        for p in batch_pred['decoding']: 
          print(p)
      else:
        break
  
  
  

