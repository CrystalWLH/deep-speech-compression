#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:53:15 2018

@author: Samuele Garda
"""

#TODO
# FIX BUFFER SIZE FOR SHUFFLE

import tensorflow as tf

def load_teacher_logits(tfrecord_logits):
  """
  Create dataset with containing teacher logits.
  
  :param:
    
  :return:
    dataset_logits (tf.data.Dataset) : dataset containing logits
  """
  
  dataset_logits = tf.data.TFRecordDataset(tfrecord_logits)
            
  dataset_logits = dataset_logits.map(parse_tfrecord_logit)
    
  return dataset_logits


def student_input_func(tfrecord_path,tfrecord_logits,vocab_size,input_channels,mode, batch_size):
  """
  Create input function for student network. Contains audio features and logits of teacher network.
  
  :param:
    tfrecord_path (str) : path to tfrecord file.
    split (str) : part of the dataset. Choiches = (`train`,`dev`,`test`)
    batch_size (int) : size of mini-batches
    input_fn (function) : input (training) function for teacher model
    teacher_model_function (function) : estimator model_fn for teacher network
    params (dict) : parameters of teacher model
    model_dir (path) : path to teacher network checkpoint
    
  """
  
  with tf.variable_scope('input'):
  
    dataset_std = load_dataset(tfrecord_path)
    
    dataset_logits = load_teacher_logits(tfrecord_logits)
    
    dataset = tf.data.Dataset.zip((dataset_std, dataset_logits))
    
    if mode == 'train':
      
      dataset = dataset.repeat()
          
    dataset = dataset.padded_batch(batch_size, padded_shapes= (([input_channels,-1], [-1]), [-1,vocab_size]),
                                               padding_values = ( ( 0. , -1), 0. ))
    
    (audio,labels), logits = dataset.make_one_shot_iterator().get_next()
    
    
    features = {'audio' : audio, 'logits' : logits}
  
  return features, labels
  
  
def load_dataset(tfrecord_path):
  """
  Load tfrecord files in dataset.
  
  :param:
    tfrecord_path (str) : path to tfrecord file.
  :return:
    dataset (tf.data.Dataset) : shuffled parsed dataset
    
  """
  
  dataset = tf.data.TFRecordDataset(tfrecord_path)
  
  dataset = dataset.map(parse_tfrecord_example)
  
#  dataset = dataset.shuffle(buffer_size= 100, seed = 42)
  
  return dataset

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
              "labels": tf.VarLenFeature(tf.int64)}
  
  parsed_features = tf.parse_single_example(proto, features)
  
  sparse_audio = parsed_features["audio"]
    
  shape = tf.cast(parsed_features["audio_shape"], tf.int32)

  dense_audio = tf.reshape(tf.sparse_to_dense(sparse_audio.indices,
                                              sparse_audio.dense_shape,
                                              sparse_audio.values),shape)
  
  sparse_trans = parsed_features["labels"]
  dense_trans = tf.sparse_to_dense(sparse_trans.indices,
                                   sparse_trans.dense_shape,
                                   sparse_trans.values)
  
  
  return dense_audio, tf.cast(dense_trans, tf.int32)


def parse_tfrecord_logit(proto):
  
  features = {"logits": tf.VarLenFeature(tf.float32),
              "shape": tf.FixedLenFeature((2,), tf.int64)}
  
  parsed_features = tf.parse_single_example(proto, features)
  
  shape = tf.cast(parsed_features["shape"], tf.int32)
  
  sparse_logits = parsed_features["logits"]
  
  dense_logits = tf.reshape(tf.sparse_to_dense(sparse_logits.indices,
                                              sparse_logits.dense_shape,
                                              sparse_logits.values),shape)
  
  return dense_logits
  
  
def teacher_input_func(tfrecord_path,input_channels, mode, batch_size):
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
  
  with tf.variable_scope('input'):
    dataset = load_dataset(tfrecord_path)  
        
    if mode == 'train':
      
      dataset = dataset.repeat()
          
    dataset = dataset.padded_batch(batch_size, padded_shapes= ([input_channels,-1],[-1]),
                                               padding_values =  (0.,-1))
          
    features,labels = dataset.make_one_shot_iterator().get_next()
      
  return features, labels


if __name__ == "__main__":
  
  next_element = teacher_input_func('./test/librispeech_tfrecords.dev',
                                   split = 'dev', batch_size = 5)
  max_len = 0
  max_trans = 0
  with tf.Session() as sess:
    for i in range(10):
      try:
        print(next_element)
        features,batch_y = sess.run(next_element)
        print(features)
        print(batch_y)
      except tf.errors.OutOfRangeError:
        print("End of dataset")
        print("Max len : {}".format(max_len))
        print("Max trans : {}".format(max_trans))
