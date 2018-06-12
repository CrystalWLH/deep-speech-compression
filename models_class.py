#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:20:27 2018

@author: Samuele Garda
"""


from abc import ABCMeta,abstractmethod
import tensorflow as tf
from utils.net import convolutional_sequence,length,clip_and_step
from utils.quantization import quant_conv_sequence,quant_clip_and_step


class AbstractW2L(object,metaclass=ABCMeta):
  
  def __init__(self,custom_op):
    
    self.custom_op = tf.load_op_library(custom_op)

  def get_seqs_length(self,logits):
    
    return length(logits)
    
  def get_by_data_format(self,features,params):
    
    with tf.variable_scope("data_format"):
      
       if params.get('data_format') == "channels_last":
      
         features = tf.transpose(features, (0, 2, 1))
        
    return features
  
  def ctc_loss(self,logits,labels,seqs_len):
    
    with tf.name_scope('ctc_loss'):
    
      sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
        
      batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                        inputs =  logits, 
                                        sequence_length = seqs_len)
      
      ctc_loss =  tf.reduce_mean(batches_ctc_loss)
      
      tf.summary.scalar('ctc_loss',ctc_loss)
      
      return ctc_loss
    
  
  def _decoding_ops(self,sparse_decoded,params):
    
    with tf.name_scope('decoder'):
                  
      dense_decoded = tf.sparse_to_dense(sparse_decoded[0].indices,
                               sparse_decoded[0].dense_shape,
                               sparse_decoded[0].values)
      
      table = tf.contrib.lookup.index_to_string_table_from_file(params.get('char2idx'),key_column_index = 0,
                                                            value_column_index = 1, delimiter = '\t',
                                                            default_value=' ')
      def init_fn(scaffold, sess):
        tf.tables_initializer().run(session = sess)
    
      scaffold = tf.train.Scaffold(init_fn=init_fn)
      
      return dense_decoded,table,scaffold
    
  def evaluate(self,mode,table,dense_decoded,labels,loss,scaffold):
    
    with tf.name_scope("evaluation"):
    
      expected_chars = table.lookup(dense_decoded)
      decoded_chars = table.lookup(tf.cast(labels,tf.int64))
      
      sparse_char_decoded = tf.contrib.layers.dense_to_sparse(decoded_chars, eos_token = 'UNK')
      sparse_char_expected = tf.contrib.layers.dense_to_sparse(expected_chars, eos_token = 'UNK')
      
      ler = tf.reduce_mean(tf.edit_distance(sparse_char_expected,sparse_char_decoded))
      
      join_expected = tf.reduce_join(expected_chars, separator = '', axis = 1)
      join_decoded = tf.reduce_join(decoded_chars, separator = '', axis = 1)
      
      split_expected = tf.string_split(join_expected)
      split_decoded = tf.string_split(join_decoded)
      
      wer = tf.reduce_mean(tf.edit_distance(split_expected,split_decoded))
    
      metrics = {"ler" : tf.metrics.mean(ler), "wer" : tf.metrics.mean(wer) }
  
    return tf.estimator.EstimatorSpec(mode=mode, loss = loss, eval_metric_ops=metrics, scaffold = scaffold)
  
  
  def predict(self,mode,table,scaffold,logits,dense_decoded):
    
    with tf.name_scope('predictions'):
      
      decoded_string = table.lookup(dense_decoded)
      
      decoded_string = tf.reduce_join(decoded_string, separator = '', axis = 1)
  
      pred = {'decoding' : decoded_string, 'logits' : logits}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred, scaffold = scaffold)
    
  def train(self,mode,params,loss):    
    
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = params.get('adam_lr'), epsilon = params.get('adam_eps'))
      if params.get('bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, params.get('clipping'))
      else:
        train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, params.get('clipping'))
        
    with tf.name_scope("visualization"):
      for g, v in grads_and_vars:
        if v.name.find("kernel") >= 0:
          tf.summary.scalar(v.name.replace(':0','_') + "gradient_norm", tf.norm(g))
      tf.summary.scalar("global_gradient_norm", glob_grad_norm)
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op)
  
  
  def decode(self,logits,seqs_len,params):
    
    if not params.get('lm') and not params.get('beam_search'):
      
      sparse_decoded,_ = tf.nn.ctc_greedy_decoder(logits, seqs_len)
      
    elif not params.get('lm') and params.get('beam_search'):
      
      sparse_decoded,_ = tf.nn.ctc_beam_search_decoder(logits,seqs_len,
                                                     beam_width=params.get('beam_width'),
                                                     top_paths=params.get('top_paths'))
      
    elif params.get('lm'):
      
      decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = self.custom_op.ctc_beam_search_decoder_with_lm(
          logits, seqs_len, beam_width=params.get('beam_width'),
          top_paths = params.get('top_paths'),
          model_path = params.get('lm_binary'),
          trie_path = params.get('lm_trie'),
          alphabet_path = params.get('lm_alphabet'),
          lm_weight = params.get('lm_weight'), 
          word_count_weight = params.get('word_count_weight'), 
          valid_word_count_weight = params.get('valid_word_count_weight'),
          merge_repeated=True)
      
      sparse_decoded = [tf.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(decoded_ixs, decoded_vals, decoded_shapes)]
      
    return sparse_decoded
  
  
  def time_major_logits(self,logits,params):
    
    with tf.name_scope('time_major'):
    
      if params.get('data_format') == 'channels_first':
        logits = tf.transpose(logits, (2,0,1))
        
      elif params.get('data_format') == 'channels_last':
        logits = tf.transpose(logits, (1,0,2))
      
    return logits
  
  
  
  def get_logits(self,features,mode,params):
    
    with tf.variable_scope("model"):
      
      logits = convolutional_sequence(inputs = features, conv_type = params.get('conv_type'),
                              filters = params.get('filters'),
                              widths = params.get('widths'),
                              strides = params.get('strides'),
                              activation = params.get('activation'),
                              data_format = params.get('data_format'),
                              dropouts = params.get('dropouts'),
                              batchnorm = params.get('bn'),
                              vocab_size = params.get('vocab_size'),
                              train = mode == tf.estimator.ModeKeys.TRAIN)
        
    return logits
  
  @abstractmethod
  def model_function(self,features,labels,mode,params):
    """
    This method should implement all the model functions
    """
    
    pass



class TeacherModel(AbstractW2L):
  
  
  def model_function(self,features,labels,mode,params):
    
    features = self.get_by_data_format(features,params)
    
    logits = self.get_logits(features,mode,params)
    
    logits = self.time_major_logits(logits,params)
    
    seqs_len = self.get_seqs_length(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,params)
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,params)
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)
    
    loss = self.ctc_loss(logits,labels,seqs_len)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,params,loss)
    
    assert mode == tf.estimator.ModeKeys.EVAL
      
    return self.evaluate(mode,table,dense_decoded,labels,loss,scaffold)
    
    

class StudentModel(AbstractW2L):
  
  def total_loss(self,ctc_loss,distillation_loss,params):
    
    with tf.name_scope("total_loss"):
    
      alpha = params.get('alpha')
      sq_temper = tf.cast(tf.square(params.get('temperature')),tf.float32)
      
  #   "Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2
  #   it is important to multiply them by T^2 when using both hard and soft targets"  
      loss =  ((1 - alpha) * ctc_loss)  +  (alpha * distillation_loss  * sq_temper)
      tf.summary.scalar('total_loss', loss)
    
    return loss

  
  def distillation_loss(self,logits,teacher_logits,params):
    
    with tf.name_scope('distillation_loss'):
    
      temperature = params.get('temperature')
      
      soft_targets = tf.nn.softmax(teacher_logits / temperature)
      soft_logits = tf.nn.softmax(logits / temperature)
      
      logits_fl = tf.reshape(soft_logits, [tf.shape(logits)[1],-1])
      st_fl = tf.reshape(soft_targets,[tf.shape(logits)[1],-1])
      
      tf.assert_equal(tf.shape(logits_fl),tf.shape(st_fl))
      
      xent_soft_targets = tf.reduce_mean(-tf.reduce_sum(st_fl * tf.log(logits_fl), axis=1))
      
      tf.summary.scalar('st_xent', xent_soft_targets)
      
    return xent_soft_targets
    
    
  def model_function(self,features,labels,mode,params):
    
    audio_features = self.get_by_data_format(features['audio'],params)
    
    teacher_logits = tf.transpose(features['logits'],(1,0,2))
    
    logits = self.get_logits(audio_features,mode,params)
    
    logits = self.time_major_logits(logits,params)
    
    seqs_len = self.get_seqs_length(logits) 
    
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,params)
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,params)
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)
      
    ctc_loss = self.ctc_loss(logits,labels,seqs_len)
    
    distillation_loss = self.distillation_loss(logits,teacher_logits,params)
    
    loss = self.total_loss(ctc_loss,distillation_loss,params)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,params,loss)
    
    assert mode == tf.estimator.ModeKeys.EVAL
      
    return self.evaluate(mode,table,dense_decoded,labels,ctc_loss,scaffold)
    
    
      
    
class QuantStudentModel(StudentModel):
  
  def get_logits(self,features,mode,params):
    
    with tf.variable_scope('model'):
    
      logits,quant_weights,original_weights = quant_conv_sequence(inputs = features,
                              conv_type = params.get('conv_type'),
                              filters = params.get('filters'),
                              widths = params.get('widths'),
                              strides = params.get('strides'),
                              activation = params.get('activation'),
                              data_format = params.get('data_format'),
                              dropouts = params.get('dropouts'),
                              batchnorm = params.get('bn'),
                              train = mode == tf.estimator.ModeKeys.TRAIN,
                              vocab_size = params.get('vocab_size'),
                              num_bits = params.get('num_bits'),
                              bucket_size = params.get('bucket_size'),
                              stochastic = params.get('stochastic'),
                              quant_last_layer = params.get('quant_last_layer'))
    
    return logits,quant_weights,original_weights
    
  def train(self,mode,params,loss,quant_weights, original_weights):
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = params.get('adam_lr'), epsilon = params.get('adam_eps'))
      if params.get('bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, quant_global_norm, orig_global_norm,orig_grads,quant_grads = quant_clip_and_step(optimizer, loss, params.get('clipping'),
                                                                                                     quant_weights, original_weights)
      else:
        train_op, quant_global_norm, orig_global_norm,orig_grads,quant_grads = quant_clip_and_step(optimizer, loss, params.get('clipping'),
                                                                                                   quant_weights, original_weights)
        
    with tf.name_scope("visualization"):
      for idx,(g_orig, g_quant) in enumerate(zip(orig_grads,quant_grads)):
        tf.summary.scalar("model/conv_layer_{}/conv/kernel_gradient_norm".format(idx), tf.norm(g_orig))
        tf.summary.scalar("model/quant_conv_layer_{}/conv/kernel_gradient_norm".format(idx), tf.norm(g_quant))
  
      tf.summary.scalar("global_gradient_norm", orig_global_norm)
      tf.summary.scalar("quant_global_gradient_norm", quant_global_norm)
      
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op) 
  
  
  def model_function(self,features,labels,mode,params):
    
    audio_features = self.get_by_data_format(features['audio'],params)
    
    teacher_logits = tf.transpose(features['logits'],(1,0,2))
    
    logits,quant_weights,original_weights = self.get_logits(audio_features,mode,params)
    
    logits = self.time_major_logits(logits,params)
    
    seqs_len = self.get_seqs_length(logits) 
    
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,params)
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,params)
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)
      
    ctc_loss = self.ctc_loss(logits,labels,seqs_len)
    
    distillation_loss = self.distillation_loss(logits,teacher_logits,params)
    
    loss = self.total_loss(ctc_loss,distillation_loss,params)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,params,loss,quant_weights,original_weights)
    
    if mode == tf.estimator.ModeKeys.EVAL:
      
      return self.evaluate(mode,table,dense_decoded,labels,ctc_loss,scaffold)
  
  