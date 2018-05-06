#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:12:41 2018

@author: Samuele Garda
"""

import argparse
from configparser import ConfigParser
import json

def parse_arguments():
  """
  Read configuration file.
  """
  
  parser = argparse.ArgumentParser(description='Run experiments for Deep ASR model compression through Teacher-Student training')
  parser.add_argument('-c', '--conf', required=True, help='Architecture configuration file')
  args = parser.parse_args()
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read(args.conf)
  return configuration


def config2params(configuration):
  
  params = {}
  params['teacher'] = {}
  params['student'] = {}
  
  
  params['data_format'] = configuration['GENERAL'].get('data_format','channels_last')
  params['save_models'] = configuration['GENERAL'].get('save_models','models')
  params['input_type'] = configuration['GENERAL'].get('input_type','ampl')
  params['conv_type'] = configuration['GENERAL'].get('input_type','conv')
  params['vocab_path'] = configuration['GENERAL'].get('vocab_path','./test/vocab.pkl')
  
  
  params['train'] = configuration['TRAIN'].get('train','teacher')
  params['activation'] = configuration['TRAIN'].get('activation','relu')
  params['batch_size'] = configuration['TRAIN'].getint('batch_size', 512)
  params['steps'] = configuration['TRAIN'].getint('steps', 10)
  params['bn'] = configuration['TRAIN'].getboolean('bn', False)
  params['temperature'] = configuration['TRAIN'].getint('temperature', 3)
  
  params['teacher']['filters'] = json.loads(configuration['TEACHER'].get('filters', 3))
  params['teacher']['widths'] = json.loads(configuration['TEACHER'].get('widths', 3))
  params['teacher']['strides'] = json.load(configuration['TEACHER'].get('strides', 3))
  
  params['student']['filters'] = json.loads(configuration['STUDENT'].get('filters', 3))
  params['student']['widths'] = json.loads(configuration['STUDENT'].get('widths', 3))
  params['student']['strides'] = json.loads(configuration['STUDENT'].get('strides', 3))
  
  return params
  
  
  
  
  
  
  
  
  
  
  

