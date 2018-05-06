#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:01:21 2018

@author: Samuele Garda
"""

import tensorflow as tf

def convolutional_sequence(inputs,conv_type,filters, widths, strides, activation, train):
  
  layer = inputs
  
  
  


def create_model_function(features, labels, mode, params):
  
  if params.get('data_format') == "channels_last":
    
    features = tf.transpose(features, [0, 2, 1])
    
  
    
    
    
  
  