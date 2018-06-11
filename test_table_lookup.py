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

config = './configs/local_test/student.config'

env_params,params = config2params(config)

features,labels = student_input_func(tfrecord_path = env_params.get('train_data'),
                                     tfrecord_logits = env_params.get('teacher_logits'),
                                     vocab_size = len(env_params.get('char2idx')),
                                     input_channels = env_params.get('input_channels'), 
                                     mode = 'eval',
                                     epochs = 1,
                                     batch_size = 1,
                                  )


table = tf.contrib.lookup.index_to_string_table_from_file('./test_lookup',key_column_index = 0, value_column_index = 1, delimiter = '\t')

sess = tf.Session()


tf.tables_initializer().run(session = sess)

#logits = tf.transpose(features['logits'], (1,0,2))
#
#seqs_len = length(logits)
#
#sparse_decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seqs_len)
#
#sparse_decoded = sparse_decoded[0]
#
#dense_decoded = tf.sparse_to_dense(sparse_decoded.indices,
#                                              sparse_decoded.dense_shape,
#                                              sparse_decoded.values)

chars_decoded = table.lookup(tf.cast(labels,tf.int64))


test_join = tf.reduce_join(chars_decoded, separator = '', axis = 1, keep_dims = False)

#print(test_join)
#
test_split = tf.string_split(test_join)
print(test_split.eval(session = sess))
