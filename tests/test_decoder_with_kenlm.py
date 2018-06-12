#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:57:01 2018

@author: Samuele Garda
"""

import tensorflow as tf
from input_funcs import student_input_func
from main import config2params
from utils.net import length 

KENLM_DECODER_OP = './data/test_knlm/libctc_decoder_with_kenlm.so'
LM_BINARY_PATH = './data/kenlm_data/lm.binary'
LM_TRIE_PATH = './data/kenlm_data/trie'
ALPHABET_PATH = './data/kenlm_data/alphabet.txt'

sess = tf.Session()

custom_op_module = tf.load_op_library(KENLM_DECODER_OP)

table = tf.contrib.lookup.index_to_string_table_from_file('./data/tfrecords_data/ctc_vocab.txt',key_column_index = 0,
                                                          value_column_index = 1, delimiter = '\t',
                                                          default_value=' ')
tf.tables_initializer().run(session = sess)

def decode_with_lm(inputs, sequence_length, beam_width=100,top_paths=1, merge_repeated=True):
  
  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = custom_op_module.ctc_beam_search_decoder_with_lm(
          inputs, sequence_length, beam_width=beam_width,
          model_path= LM_BINARY_PATH,
          trie_path=LM_TRIE_PATH,
          alphabet_path=ALPHABET_PATH,
          lm_weight=1.75, 
          word_count_weight=1.00, 
          valid_word_count_weight=1.00,
          top_paths=top_paths,
          merge_repeated=merge_repeated)
  
  return ([tf.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(decoded_ixs, decoded_vals, decoded_shapes)],log_probabilities)


config = './configs/local_test/student.config'

print("Load logits")
env_params,params = config2params(config)

features,labels = student_input_func(tfrecord_path = env_params.get('train_data'),
                                     tfrecord_logits = env_params.get('teacher_logits'),
                                     vocab_size = env_params.get('vocab_size'),
                                     input_channels = env_params.get('input_channels'), 
                                     mode = 'eval',
                                     epochs = 1,
                                     batch_size = 5,
                                  )

logits = tf.transpose(features['logits'], (1,0,2))
seqs_len = length(logits)

print("Decode with LM")
sparse_decoded, _ = decode_with_lm(logits, seqs_len, merge_repeated=False, beam_width=1024)

dense_decoded = tf.sparse_to_dense(sparse_decoded[0].indices,
                                              sparse_decoded[0].dense_shape,
                                              sparse_decoded[0].values)

print("From ids to chars")
expected_chars = table.lookup(dense_decoded)

join_expected = tf.reduce_join(expected_chars, separator = '', axis = 1)

print(join_expected.eval(session = sess))

sess.close()

