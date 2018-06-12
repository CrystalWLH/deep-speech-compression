#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:08:21 2018

@author: Samuele Garda
"""

import tensorflow as tf
from input_funcs import student_input_func
from main import config2params
from utils.net import length 



sess = tf.Session()
config = './configs/local_test/student.config'

env_params,params = config2params(config)

features,labels = student_input_func(tfrecord_path = env_params.get('train_data'),
                                     tfrecord_logits = env_params.get('teacher_logits'),
                                     vocab_size = len(env_params.get('char2idx')),
                                     input_channels = env_params.get('input_channels'), 
                                     mode = 'eval',
                                     epochs = 1,
                                     batch_size = 5,
                                  )




table = tf.contrib.lookup.index_to_string_table_from_file('./test_lookup',key_column_index = 0,
                                                          value_column_index = 1, delimiter = '\t',
                                                          default_value=' ')
tf.tables_initializer().run(session = sess)

logits = tf.transpose(features['logits'], (1,0,2))

seqs_len = length(logits)

sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)

dense_decoded = tf.sparse_to_dense(sparse_decoded[0].indices,
                                              sparse_decoded[0].dense_shape,
                                              sparse_decoded[0].values)

expected_chars = table.lookup(dense_decoded)
decoded_chars = table.lookup(tf.cast(labels,tf.int64))

print(expected_chars.eval(session = sess))
print(decoded_chars.eval(session = sess))


sparse_char_decoded = tf.contrib.layers.dense_to_sparse(decoded_chars, eos_token = 'UNK')
sparse_char_expected = tf.contrib.layers.dense_to_sparse(expected_chars, eos_token = 'UNK')

test_ler = tf.edit_distance(sparse_char_expected,sparse_char_decoded)

print(test_ler.eval(session = sess))

join_expected = tf.reduce_join(expected_chars, separator = '', axis = 1)
join_decoded = tf.reduce_join(decoded_chars, separator = '', axis = 1)

print(join_expected.eval(session = sess))
print(join_decoded.eval(session = sess))

split_expected = tf.string_split(join_expected)
split_decoded = tf.string_split(join_decoded)

print(split_expected.eval(session = sess))
print(split_decoded.eval(session = sess))


test_wer = tf.edit_distance(split_expected,split_decoded)

print(test_wer.eval(session = sess))



sess.close()
