#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:53:15 2018

@author: Samuele Garda
"""

#TODO
# EVAL AND PREDICT NOT WORKING! FFS 

import tensorflow as tf

def parse_tfrecord_example(proto):
  """
  Used to parse examples in tf.data.Dataset. 
  Each example is mapped in its feature representation and its labels (int encoded transcription of audio).
  
  :param:
    proto (tf.Tensor) : element stored in tf.data.TFRecordDataset
    
  :return:
    dense_audio (tf.Tensor) : audio
    dense_trans (tf.Tensor) : encoded transcription
  
  """
  
  features = {"audio": tf.VarLenFeature(tf.float32),
              "audio_shape": tf.FixedLenFeature((2,), tf.int64),
              "seq_len" :  tf.FixedLenFeature((), tf.int64),
              "labels": tf.VarLenFeature(tf.int64)}
  
  parsed_features = tf.parse_single_example(proto, features)
  
  sparse_audio = parsed_features["audio"]
  
  seq_len = tf.cast(parsed_features['seq_len'], tf.int32)
  
  shape = tf.cast(parsed_features["audio_shape"], tf.int32)

  dense_audio = tf.reshape(tf.sparse_to_dense(sparse_audio.indices,
                                              sparse_audio.dense_shape,
                                              sparse_audio.values),shape)
  
  sparse_trans = parsed_features["labels"]
  dense_trans = tf.sparse_to_dense(sparse_trans.indices,
                                   sparse_trans.dense_shape,
                                   sparse_trans.values)
  
  
  batch = {'audio' : dense_audio, 'seq_len' : seq_len, 'labels' : tf.cast(dense_trans, tf.int32) }
  
  return batch 


def expand(x):
  '''
  Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar  to a vector of length 1
  :param x:
  :return:
  '''
  x['seq_len'] = tf.expand_dims(x['seq_len'],0)
  return x 

def deflate(x):
  '''
  Undo Hack. We undo the expansion we did in expand
  '''
  x['seq_len'] = tf.squeeze(x['seq_len'])
  return x


def model_input_func_tfr(tfrecord_path, split, shuffle, batch_size):
  """
  Create dataset instance from tfrecord file. It prefetches mini-batch. If is train split dataset is repeated.
  
  :param:
    tfrecord_path (str) : path to tfrecord file.
    split (str) : part of the dataset. Choiches = (`train`,`dev`,`test`)
    shuffle (int) : buffer size for shuffle operation
    batch_size (int) : size of mini-batches
    
  :return:
    minibatch (batch_features,batch_labels) : minibatch (features, labels)
  """
    
  dataset = tf.data.TFRecordDataset(tfrecord_path)
  
  dataset = dataset.map(parse_tfrecord_example)
  
  dataset = dataset.shuffle(buffer_size=shuffle)
  
  dataset = dataset.map(expand)
    
  if split == 'train':
    
    dataset = dataset.repeat()
        
  dataset = dataset.padded_batch(batch_size, padded_shapes= {"audio" : [257,-1], "seq_len" : 1, "labels" : [-1]},
                                             padding_values = {'audio' : 0. , 'seq_len' : 1,'labels' : -1})
  dataset = dataset.map(deflate)
      
  features = dataset.make_one_shot_iterator().get_next()
  
  labels = features.pop('labels')
      
  return features, labels


if __name__ == "__main__":
  
  next_element = model_input_func_tfr('./test/librispeech_tfrecords.dev', shuffle = 10,
                                   split = 'dev', batch_size = 5)
  
  max_len = 0
  max_trans = 0
  with tf.Session() as sess:
    for i in range(102):
      try:
        features,batch_y = sess.run(next_element)
        print(features['seq_len'])
        print(features['audio'])
        print(batch_y)
      except tf.errors.OutOfRangeError:
        print("End of dataset")
        print("Max len : {}".format(max_len))
        print("Max trans : {}".format(max_trans))
