#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:08:21 2018

@author: Samuele Garda
"""

import numpy as np
import tensorflow as tf
from input_funcs import teacher_input_func
from main import config2params
from utils.net import length
from utils.net import convolutional_sequence



sess = tf.Session()
config = './configs/local_test/student.config'

env_params,params = config2params(config)

features,labels = teacher_input_func(tfrecord_path = env_params.get('train_data'),                            
                                     input_channels = 39,
                                     mode = 'eval',
                                     epochs = 1,
                                     batch_size = 3,
                                  )


if params.get('data_format') == "channels_last":
      
  features = tf.transpose(features, (0, 2, 1))

logits = convolutional_sequence(inputs = features, conv_type = params.get('conv_type'),
                            filters = params.get('filters'),
                            widths = params.get('widths'),
                            strides = params.get('strides'),
                            vocab_size = params.get('vocab_size'),
                            activation = params.get('activation'),
                            data_format = params.get('data_format'),
                            dropouts = params.get('dropouts'),
                            batchnorm = params.get('bn'),
                            train = False)

if params.get('data_format') == 'channels_first':
  logits = tf.transpose(logits, (2,0,1))
  
elif params.get('data_format') == 'channels_last':
  logits = tf.transpose(logits, (1,0,2))

seqs_len = length(logits)

sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)

dense_decoded = tf.sparse_to_dense(sparse_decoded[0].indices,
                                              sparse_decoded[0].dense_shape,
                                              sparse_decoded[0].values)

sess.run(tf.global_variables_initializer())

table = tf.contrib.lookup.index_to_string_table_from_file(params.get('char2idx'),key_column_index = 0,
                                                          value_column_index = 1, delimiter = '\t',
                                                          default_value=' ')
tf.tables_initializer().run(session = sess)

expected_chars = table.lookup(dense_decoded)
join_expected = tf.reduce_join(expected_chars, separator = '', axis = 1)

test = dense_decoded.eval(session = sess)

#test_seqs_len = length(test)

print(test.shape)
#print(test_seqs_len.eval(session = sess))
#


#
#logits = tf.transpose(features['logits'], (1,0,2))
#
#
#
#
#
#dense_decoded = tf.sparse_to_dense(sparse_decoded[0].indices,
#                                              sparse_decoded[0].dense_shape,
#                                              sparse_decoded[0].values)
#
#
#decoded_chars = table.lookup(tf.cast(labels,tf.int64))
#
#print(expected_chars.eval(session = sess))
#print(decoded_chars.eval(session = sess))
#
#
#sparse_char_decoded = tf.contrib.layers.dense_to_sparse(decoded_chars, eos_token = 'UNK')
#sparse_char_expected = tf.contrib.layers.dense_to_sparse(expected_chars, eos_token = 'UNK')
#
#test_ler = tf.edit_distance(sparse_char_expected,sparse_char_decoded)
#
#print(test_ler.eval(session = sess))
#
#join_expected = tf.reduce_join(expected_chars, separator = '', axis = 1)
#join_decoded = tf.reduce_join(decoded_chars, separator = '', axis = 1)
#
#
#print(join_decoded.eval(session = sess))
#
#split_expected = tf.string_split(join_expected)
#split_decoded = tf.string_split(join_decoded)
#
#print(split_expected.eval(session = sess))
#print(split_decoded.eval(session = sess))
#
#
#test_wer = tf.edit_distance(split_expected,split_decoded)
#
#print(test_wer.eval(session = sess))
#
#
#
#sess.close()
