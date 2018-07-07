#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:20:27 2018

@author: Samuele Garda
"""


from abc import ABCMeta,abstractmethod
import tensorflow as tf
from utils.net import _conv_seq,convolutional_sequence,length,clip_and_step
from utils.quantization import quant_conv_sequence,quant_clip_and_step


class AbstractW2L(object,metaclass=ABCMeta):
  """
  Abstract class for W2L model.
  """
  
  def __init__(self,custom_op):
    """
    Create new object. If `custom_op` is None decoding with KenLm won't be available
    
    :param:
      custom_op (str) : path to compiled op `ctc_beam_search_decoder_with_lm`
    
    """
    
    self.custom_op = tf.load_op_library(custom_op) if custom_op else None

  def get_seqs_length(self,logits):
    """
    Get lengths of logits (in time major format).
    
    :param:
      logits (tf.Tensor) : logits
    
    :return:
      True length of each example in batch
    """
    
    return length(logits)
    
  def get_by_data_format(self,features, data_format):
    """
    Ìnput expected in `channels_first` format. Transpose input if data_format is `channels_last`.
    
    :param:
      features (tf.Tensor) : feautres [batch,channels,max_time]
      data_format (str) : data format specification
      
    :return:
      features (tf.Tensor) : (transposed if necessary) features 
    
    """
    
    with tf.variable_scope("data_format"):
      
       if data_format == "channels_last":
      
         features = tf.transpose(features, (0, 2, 1))
        
    return features
  
  def ctc_loss(self,logits,labels,seqs_len):
    """
    Compute CTC loss.
    
    :params:
      logits (tf.Tensor) : logits
      labels (tf.Tensor) : dense Tensor labels
      seqs_len (tf.Tensor) : true length of each example in batch
      
    :return:
      ctc_loss (scalar) : computed loss
      
    """
    
    with tf.name_scope('ctc_loss'):
    
      sparse_labels = tf.contrib.layers.dense_to_sparse(labels, eos_token = -1)
        
      batches_ctc_loss = tf.nn.ctc_loss(labels = sparse_labels,
                                        inputs =  logits, 
                                        sequence_length = seqs_len)
      
      ctc_loss =  tf.reduce_mean(batches_ctc_loss)
      
      tf.summary.scalar('ctc_loss',ctc_loss)
      
      return ctc_loss
    
  
  def _decoding_ops(self,sparse_decoded,lookup_path):
    """
    Initialiaze decoding ops. Sparse tensor containing output is transformed back to dense.
    A lookup table is initialized for getting characters with `tf.train.Scaffold`.
    
    :params:
      sparse_decoded (tf.SparseTensor) : sparse decoded logits
      lookup_path (str) : path to lookup file. MUST be in format : <value>\t<character>\n
    :return:
      dense_decoded (tf.Tensor) : dense decoded logits
      table (tf.HashTable) : lookup table
      scaffold (tf.Scaffold) : op that initialize `table` within session
      
    """
    
    with tf.name_scope('decoder'):
                  
      dense_decoded = tf.sparse_to_dense(sparse_decoded[0].indices,
                               sparse_decoded[0].dense_shape,
                               sparse_decoded[0].values)
      
      table = tf.contrib.lookup.index_to_string_table_from_file(lookup_path,key_column_index = 0,
                                                            value_column_index = 1, delimiter = '\t',
                                                            default_value=' ')
      def init_fn(scaffold, sess):
        tf.tables_initializer().run(session = sess)
    
      scaffold = tf.train.Scaffold(init_fn=init_fn)
      
      return dense_decoded,table,scaffold
    
  def evaluate(self,mode,table,dense_decoded,labels,loss,scaffold):
    """
    Evaluate model with LER and WER.
    
    :param:
      mode (str) : mode (must be `eval`)
      table (tf.HashTable) : lookup table
      dense_decoded (tf.Tensor) : dense decoded logits
      labels (tf.Tensor) : labels in dense format
      loss (scalar) : loss
      scaffold (tf.Scaffold) : op that initialize `table` within session
    
    :return:
      tf.estimator.EstimatorSpec for evaluation mode
      
    """
    
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
    """
    Make predictions with model. 
    
    :param:
      mode (str) : mode (must be `predict`)
      table (tf.HashTable) : lookup table
      scaffold (tf.Scaffold) : op that initialize `table` within session
      logits (tf.Tensor) : logits
      dense_decoded (tf.Tensor) : dense decoded logits
      
    :return:
      tf.estimator.EstimatorSpec for evaluation mode
    """
    
    with tf.name_scope('predictions'):
      
      decoded_string = table.lookup(dense_decoded)
      
      decoded_string = tf.reduce_join(decoded_string, separator = '', axis = 1)
      
#      decoded_string = tf.regex_replace(decoded_string,pattern = 'VVV*',rewrite = '')
  
      pred = {'decoding' : decoded_string, 'logits' : logits}
      
    return tf.estimator.EstimatorSpec(mode = mode, predictions=pred, scaffold = scaffold)
    
  def train(self,mode,lr,eps,clipping,bn,loss):
    """
    Train model. Add gradient visualization.
    
    :params:
      mode (str) : mode (must be `train`)
      lr (int) : learning rate
      eps (int) : second Adam parameter
      clipping (int) : if magnitude global gradient exceeds this value apply gradient clipping. 0 for disabling
      bn (bool) : if batch normalization layers were used
      loss (scalar) : loss to be minimized
    :return:
      tf.estimator.EstimatorSpec for evaluation mode
    """
    
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = lr, epsilon = eps)
      if bn:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, clipping)
      else:
        train_op, grads_and_vars, glob_grad_norm = clip_and_step(optimizer, loss, clipping)
        
    with tf.name_scope("visualization"):
      for g, v in grads_and_vars:
        if v.name.find("kernel") >= 0:
          tf.summary.scalar(v.name.replace(':0','_') + "gradient_norm", tf.norm(g))
      tf.summary.scalar("global_gradient_norm", glob_grad_norm)
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op)
  
  
  def decode(self,logits,seqs_len,lm, beam_search,beam_width,top_paths,
             lm_binary,lm_trie,lm_alphabet,lm_weight,word_count_weight,valid_word_count_weight ):
    """
    Decode from logits.
    
    :params:
      logits (tf.Tensor) : 3D time major logits
      seqs_len (tf.Tensor) : true length of each example in batch
      lm (bool) : use KenLM
      beam_search (bool) : use Beam Search decoder
      beam_width (int) : width of beams for Beam Search
      top_paths (int) : An int scalar >= 0, <= beam_width (controls output size)
      lm_binary (str) : path to binary file of KenLM model
      lm_trie (str) : path to trie storing vocabulary
      lm_alphabet (str) : path to file storing alphabet
      lm_weight (int) : KenLM parameter
      word_count_weight (int) : weight for word counts
      valid_word_count_weight (int) : weight for valid word encountered
      
    :return:
      sparse_decoded (tf.SparseTensor) : sparse tensor containing decoding for logits
    """
    
    if not lm and not beam_search:
      
      sparse_decoded,_ = tf.nn.ctc_greedy_decoder(logits, seqs_len)
      
    elif not lm and beam_search:
      
      sparse_decoded,_ = tf.nn.ctc_beam_search_decoder(logits,seqs_len,
                                                     beam_width=beam_width,
                                                     top_paths=top_paths,
                                                     merge_repeated=False)
      
    elif lm:
      
      if not self.custom_op:
        
        raise ValueError("You need to specify the path to the compiled op if you want to use `ctc_beam_search_decoder_with_lm`")
        
      else:
      
        decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = self.custom_op.ctc_beam_search_decoder_with_lm(
            logits, seqs_len, 
            beam_width=beam_width,
            top_paths = top_paths,
            model_path = lm_binary,
            trie_path = lm_trie,
            alphabet_path = lm_alphabet,
            lm_weight = lm_weight,
            word_count_weight = word_count_weight,
            valid_word_count_weight = valid_word_count_weight,
            merge_repeated=False)
        
        sparse_decoded = [tf.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(decoded_ixs, decoded_vals, decoded_shapes)]
      
    return sparse_decoded
  
  
  def time_major_logits(self,logits,data_format):
    """
    get logits in time major format, i.e. : [max_time,batch,channels]
    
    :params:
      logits (tf.Tensor) : 3D logits
      data_format (str) : data format specification
    """
    
    with tf.name_scope('time_major'):
    
      if data_format == 'channels_first':
        logits = tf.transpose(logits, (2,0,1))
        
      elif data_format == 'channels_last':
        logits = tf.transpose(logits, (1,0,2))
      
    return logits
  
  
  
  def get_logits(self,features,mode,conv_type,filters,widths,strides,activation,data_format,dropouts,bn,vocab_size):
    """
    Model core. Computes logits via a convolutional sequence.
    
    :params:
      Refer to `utils.convolution_sequence` for documentation on parameters.
    :return:
      logits (tf.Tensor) : 3D logits
  
    """
    
    with tf.variable_scope("model"):
      
      logits = convolutional_sequence(inputs = features,
                              conv_type =conv_type,
                              filters = filters,
                              widths = widths,
                              strides = strides,
                              activation = activation,
                              data_format = data_format,
                              dropouts =dropouts,
                              batchnorm = bn,
                              vocab_size = vocab_size,
                              train = mode)
        
    return logits
  
  @abstractmethod
  def model_function(self,features,labels,mode,params):
    """
    This method should implement the actual model logic
    """
    
    pass



class TeacherModel(AbstractW2L):
  
  def model_function(self,features,labels,mode,params):
    """
    Model function for training,evaluating and predicting from `features`.
    
    :params:
      features (tf.Tensor) : input batch
      labels (tf.Tensor) : labels batch
      mode (str) : mode
      params (dict) : all parameters of model
      
    """
    
    
    features = self.get_by_data_format(features,data_format = params.get('data_format'))
    
    logits = self.get_logits(features,mode,
                             conv_type = params.get('conv_type'), 
                             filters = params.get('filters'),
                             widths = params.get('widths'),
                             strides = params.get('strides'),
                             activation = params.get('activation'),
                             data_format = params.get('data_format'),
                             dropouts = params.get('dropouts'),
                             bn = params.get('bn'),
                             vocab_size = params.get('vocab_size'))
    
    logits = self.time_major_logits(logits,data_format = params.get('data_format'))
    
    seqs_len = self.get_seqs_length(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,
                                   lm = params.get('lm'), 
                                   beam_search= params.get('beam_search'),
                                   beam_width= params.get('beam_width'),
                                   top_paths= params.get('top_paths'),
                                   lm_binary= params.get('lm_binary'),
                                   lm_trie= params.get('lm_trie'),
                                   lm_alphabet= params.get('lm_alphabet'),
                                   lm_weight= params.get('lm_weight'),
                                   word_count_weight= params.get('word_count_weight'),
                                   valid_word_count_weight= params.get('valid_word_count_weight') )
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,lookup_path = params.get('char2idx'))
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)
    
    loss = self.ctc_loss(logits,labels,seqs_len)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,
                        lr = params.get('adam_lr'),
                        eps = params.get('adam_eps'),
                        clipping = params.get('clipping'),
                        bn = params.get('bn'),
                        loss = loss)
    
    assert mode == tf.estimator.ModeKeys.EVAL
      
    return self.evaluate(mode,table,dense_decoded,labels,loss,scaffold)
    
    

class StudentModel(AbstractW2L):
  
  def total_loss(self,ctc_loss,distillation_loss,alpha,temperature):
    """
    Compute weighted average of ctc loss and distillation loss.
    
    :params:
      ctc_loss (scalar) : CTC loss
      distillation_loss (scalar) : cross entropy with teacher logits
      alpha (int) : weighting factor
      temperature (int) : temperature at which distillation was performed
    :return:
      loss : weighted total loss
    """
    
    with tf.name_scope("total_loss"):
    
      alpha = alpha
      sq_temper = tf.cast(tf.square(temperature),tf.float32)
      
  #   "Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2
  #   it is important to multiply them by T^2 when using both hard and soft targets"  
      loss =  ((1 - alpha) * ctc_loss)  +  (alpha * distillation_loss  * sq_temper)
      tf.summary.scalar('total_loss', loss)
    
    return loss

  
  def distillation_loss(self,logits,teacher_logits,temperature):
    """
    Compute cross entropy between teacher logits and student logits raised to specified temperature.
    
    :params:
      logits (tf.Tensor) : 3D time major student logits
      teacher_logits (tf.Tensor) : 3D time major teacher logits
      temperature (int) : at which temperature perform distillation.
    
    :return:
      xent_soft_targets (scalar) : cross entropy between teacher logits and student logits 
      
    """
    
    with tf.name_scope('distillation_loss'):
      
      soft_targets = tf.nn.softmax(teacher_logits / temperature)
      soft_logits = tf.nn.softmax(logits / temperature)
      
      logits_fl = tf.reshape(soft_logits, [tf.shape(logits)[1],-1])
      st_fl = tf.reshape(soft_targets,[tf.shape(logits)[1],-1])
      
      tf.assert_equal(tf.shape(logits_fl),tf.shape(st_fl))
      
      xent_soft_targets = tf.reduce_mean(-tf.reduce_sum(st_fl * tf.log(logits_fl), axis=1))
      
      tf.summary.scalar('st_xent', xent_soft_targets)
      
    return xent_soft_targets
    
    
  def model_function(self,features,labels,mode,params):
    """
    Model function for training,evaluating and predicting from `features` with Teacher-Student training.
    
    :params:
      features (tf.Tensor) : input batch
      labels (tf.Tensor) : labels batch
      mode (str) : mode
      params (dict) : all parameters of model
      
    """

    
    audio_features = self.get_by_data_format(features['audio'],data_format = params.get('data_format'))
    
    teacher_logits = self.time_major_logits(features['guide'],data_format = 'channels_last')
    
    logits = self.get_logits(audio_features,mode,
                             conv_type = params.get('conv_type'), 
                             filters = params.get('filters'),
                             widths = params.get('widths'),
                             strides = params.get('strides'),
                             activation = params.get('activation'),
                             data_format = params.get('data_format'),
                             dropouts = params.get('dropouts'),
                             bn = params.get('bn'),
                             vocab_size = params.get('vocab_size'))
    
    logits = self.time_major_logits(logits,data_format = params.get('data_format'))
    
    seqs_len = self.get_seqs_length(logits) 
    
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,
                                   lm = params.get('lm'), 
                                   beam_search= params.get('beam_search'),
                                   beam_width= params.get('beam_width'),
                                   top_paths= params.get('top_paths'),
                                   lm_binary= params.get('lm_binary'),
                                   lm_trie= params.get('lm_trie'),
                                   lm_alphabet= params.get('lm_alphabet'),
                                   lm_weight= params.get('lm_weight'),
                                   word_count_weight= params.get('word_count_weight'),
                                   valid_word_count_weight= params.get('valid_word_count_weight') )
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,lookup_path = params.get('char2idx'))
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)
      
    ctc_loss = self.ctc_loss(logits,labels,seqs_len)
    
    distillation_loss = self.distillation_loss(logits,teacher_logits,temperature = params.get('temperature'))
    
    loss = self.total_loss(ctc_loss, distillation_loss, alpha = params.get('alpha'), temperature = params.get('temperature'))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,
                        lr = params.get('adam_lr'),
                        eps = params.get('adam_eps'),
                        clipping = params.get('clipping'),
                        bn = params.get('bn'),
                        loss = loss)
    
    assert mode == tf.estimator.ModeKeys.EVAL
      
    return self.evaluate(mode,table,dense_decoded,labels,ctc_loss,scaffold)
    


class FitNet(StudentModel):
  
  def hint_format(self,hint, data_format):
    """
    Hint format it's `channels_last` (batch,time,channels). If using `channels_first` hints are transposed.
    
    :param:
      hint (tf.Tensor) : tensor containing hidden representations from Teacher Network
      data_format (str) : data format specification
      
    :return:
      hint (tf.Tensor) : eventually transposed tensor
    """
    
    with tf.variable_scope('hint_format'):
      
      if data_format == 'channels_first':
        
        hint = tf.transpose(hint, [0,2,1])
      
    return hint
  
  def fitnet_loss(self,guided,hint,data_format,activation):
    """
    Compute weighted squared error between the guided layer and the hint layer. 
    If the dimensions of the guided layer and the hint layer do not match, a convolutional regressor is applied to overcome this issue.
    
    :math:`Loss_{fitent} = \\frac{1}{2}|| guided - r(hint * W^{i})||^{2}`
    
    otherwise the loss is computed directly.
    
    :params:
      
      guided (tf.Tensor) : the guided layer. Activated output of n-th convolution in Student model
      hint (tf.Tensor) : the guiding layer. Activated output of n-th convolution in Teacher model
      data_format (str) : data format specification
      activation (tf function) : activation function if convolutional regressor is applied
      
    :return:
      
      loss (scalar) : fitnet loss
    """
    
    with tf.name_scope("fitnet_loss"):
      
      channels_axis = 1 if data_format == "channels_first" else -1
      
      guided_channels = guided.get_shape().as_list()[channels_axis]
      hint_channels = hint.get_shape().as_list()[channels_axis]
      
      if guided_channels != hint_channels:
        
        print("Dimensions are not the same. Using Conv Regressor")
        
        with tf.variable_scope("fitnet_regressor"):
        
          guided = tf.layers.conv1d(inputs = guided,
                                      filters=hint_channels,
                                      kernel_size= 1,
                                      strides = 1,
                                      activation = activation,
                                      padding = 'same',
                                      use_bias=False,
                                      data_format = data_format,
                                      name = 'conv' )
          
      else:
        
        print("Equal dimensions, no need for regressor")
          
      
      guided_fl = tf.reshape(guided,[tf.shape(hint)[0],-1])
      hint_fl = tf.reshape(hint,[tf.shape(hint)[0],-1])
      
      tf.assert_equal(tf.shape(guided_fl),tf.shape(hint_fl))
      
      loss = tf.losses.mean_squared_error(hint_fl,guided_fl)
      
      tf.summary.scalar('fitnet_loss',loss)
    
    return loss
  
  
  def get_guided_layer(self,features,mode,conv_type,filters,widths,strides,activation,data_format,dropouts,bn,up_to):
    
    with tf.variable_scope('model'):
    
      guided = _conv_seq(inputs = features,
                         conv_type =conv_type,
                         filters = filters[:up_to],
                         widths = widths[:up_to],
                         strides = strides[:up_to],
                         activation = activation,
                         data_format = data_format,
                         dropouts =dropouts[:up_to],
                         batchnorm = bn,
                         train = mode,
                         hist = True)
    
    return guided
    
    
  
  def model_function(self,features,labels,mode,params):
    """
    Model function for training,evaluating and predicting from `features` with Teacher-Student training.
    
    :params:
      features (tf.Tensor) : input batch
      labels (tf.Tensor) : labels batch
      mode (str) : mode
      params (dict) : all parameters of model
      
    """

    
    audio_features = self.get_by_data_format(features['audio'],data_format = params.get('data_format'))
    
    
    if params.get('stage') == 1:
      
      print("I'm in stage 1 : train with HintLoss")
      
      hint = self.hint_format(features['guide'], data_format = params.get('data_format'))
      
      
      guided = self.get_guided_layer(audio_features,mode,
                               conv_type = params.get('conv_type'), 
                               filters = params.get('filters'),
                               widths = params.get('widths'),
                               strides = params.get('strides'),
                               activation = params.get('activation'),
                               data_format = params.get('data_format'),
                               dropouts = params.get('dropouts'),
                               bn = params.get('bn'),
                               up_to = params.get('guided')
                               )
      
      loss = self.fitnet_loss(guided,hint,
                              data_format = params.get('data_format'),
                              activation = params.get('activation'))
      
      

    
    elif params.get('stage') == 2:
      
      print("I'm in stage 2 : train with DL")
    
      teacher_logits = self.time_major_logits(features['guide'],data_format = 'channels_last')
    
      logits = self.get_logits(audio_features,mode,
                             conv_type = params.get('conv_type'), 
                             filters = params.get('filters'),
                             widths = params.get('widths'),
                             strides = params.get('strides'),
                             activation = params.get('activation'),
                             data_format = params.get('data_format'),
                             dropouts = params.get('dropouts'),
                             bn = params.get('bn'),
                             vocab_size = params.get('vocab_size'))
    
      logits = self.time_major_logits(logits,data_format = params.get('data_format'))
    
      seqs_len = self.get_seqs_length(logits) 
    
      
      ctc_loss = self.ctc_loss(logits,labels,seqs_len)
    
      distillation_loss = self.distillation_loss(logits,teacher_logits,temperature = params.get('temperature'))
    
      loss = self.total_loss(ctc_loss, distillation_loss, alpha = params.get('alpha'), temperature = params.get('temperature'))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,
                        lr = params.get('adam_lr'),
                        eps = params.get('adam_eps'),
                        clipping = params.get('clipping'),
                        bn = params.get('bn'),
                        loss = loss)
      
      
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,
                                   lm = params.get('lm'), 
                                   beam_search= params.get('beam_search'),
                                   beam_width= params.get('beam_width'),
                                   top_paths= params.get('top_paths'),
                                   lm_binary= params.get('lm_binary'),
                                   lm_trie= params.get('lm_trie'),
                                   lm_alphabet= params.get('lm_alphabet'),
                                   lm_weight= params.get('lm_weight'),
                                   word_count_weight= params.get('word_count_weight'),
                                   valid_word_count_weight= params.get('valid_word_count_weight') )
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,lookup_path = params.get('char2idx'))
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)

    
    assert mode == tf.estimator.ModeKeys.EVAL
      
    return self.evaluate(mode,table,dense_decoded,labels,ctc_loss,scaffold)
      
      
    
class QuantStudentModel(StudentModel):
 
  def get_logits(self,features,mode,conv_type,filters,widths,strides,activation,data_format,dropouts,batchnorm,vocab_size,
                 num_bits,bucket_size,stochastic):
    """
    Model core. Computes logits via a QUANTIZED convolutional sequence.
    
    :params:
      Refer to `quantization.quant_conv_sequence` for documentation on parameters.
    :return:
      logits (tf.Tensor) : 3D logits
      quant_weights (list) : list of tf.Tensors (quantized weights)
      original_weights (list) : list of tf.Variables (original weights)
    """

    
    with tf.variable_scope('model'):
    
      logits,quant_weights,original_weights = quant_conv_sequence(inputs = features,                                  
                              conv_type =conv_type,
                              filters = filters,
                              widths = widths,
                              strides = strides,
                              activation = activation,
                              data_format = data_format,
                              dropouts = dropouts,
                              batchnorm = batchnorm,
                              vocab_size = vocab_size,
                              train = mode,
                              num_bits = num_bits,
                              bucket_size = bucket_size,
                              stochastic = stochastic,
                              )
    
    return logits,quant_weights,original_weights
    
  def train(self,mode,loss,quant_weights, original_weights,lr,eps,clipping,bn):
    """
    Train model with QuantizedDistillation.
    
    :params:
      mode (str) : mode (must be `train`)
      loss (scalar) : loss to be minimized
      quant_weights (list) : list of tf.Tensors (quantized weights)
      original_weights (list) : list of tf.Variables (original weights)
      lr (int) : learning rate
      eps (int) : second Adam parameter
      clipping (int) : if magnitude global gradient exceeds this value apply gradient clipping. 0 for disabling
      bn (bool) : if batch normalization layers were used
      
    """
    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(learning_rate = lr, epsilon = eps)
      if bn:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op, quant_global_norm, orig_global_norm, orig_grads = quant_clip_and_step(optimizer, loss, clipping,
                                                                                                     quant_weights, original_weights)
      else:
        train_op, quant_global_norm, orig_global_norm, orig_grads = quant_clip_and_step(optimizer, loss, clipping,
                                                                                                   quant_weights, original_weights)
    
    
    with tf.name_scope("visualization"):
      kernels = 0
      for g_orig in orig_grads:
        
        if len(g_orig.get_shape().as_list()) < 2:
          pass
        else:
          if 'logits' in g_orig.name:
            tf.summary.scalar("model/logits/kernel_gradient_norm", tf.norm(g_orig))
          else:
            tf.summary.scalar("model/conv_layer_{}/conv/kernel_gradient_norm".format(kernels), tf.norm(g_orig))
            kernels += 1
  
      tf.summary.scalar("global_gradient_norm", orig_global_norm)
      tf.summary.scalar("quant_global_gradient_norm", quant_global_norm)
      
  
      
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op) 
  
  
  def model_function(self,features,labels,mode,params):
    """
    Model function for training,evaluating and predicting from `features` with Quantized Distillation.
    
    :params:
      features (tf.Tensor) : input batch
      labels (tf.Tensor) : labels batch
      mode (str) : mode
      params (dict) : all parameters of model
      
    """
    
    audio_features = self.get_by_data_format(features['audio'],data_format = params.get('data_format'))
    
    teacher_logits = self.time_major_logits(features['guide'],data_format = 'channels_last')
    
    logits,quant_weights,original_weights = self.get_logits(audio_features,mode,
                             conv_type = params.get('conv_type'), 
                             filters = params.get('filters'),
                             widths = params.get('widths'),
                             strides = params.get('strides'),
                             activation = params.get('activation'),
                             data_format = params.get('data_format'),
                             dropouts = params.get('dropouts'),
                             batchnorm = params.get('bn'),
                             vocab_size = params.get('vocab_size'),
                             num_bits = params.get('num_bits'),
                             bucket_size = params.get('bucket_size'),
                             stochastic = params.get('stochastic')
                             )
    
    
    logits = self.time_major_logits(logits,data_format = params.get('data_format'))
    
    seqs_len = self.get_seqs_length(logits) 
    
    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
      
      sparse_decoded = self.decode(logits,seqs_len,
                                   lm = params.get('lm'), 
                                   beam_search= params.get('beam_search'),
                                   beam_width= params.get('beam_width'),
                                   top_paths= params.get('top_paths'),
                                   lm_binary= params.get('lm_binary'),
                                   lm_trie= params.get('lm_trie'),
                                   lm_alphabet= params.get('lm_alphabet'),
                                   lm_weight= params.get('lm_weight'),
                                   word_count_weight= params.get('word_count_weight'),
                                   valid_word_count_weight= params.get('valid_word_count_weight') )
      
      dense_decoded,table,scaffold = self._decoding_ops(sparse_decoded,lookup_path = params.get('char2idx'))
      
    if mode == tf.estimator.ModeKeys.PREDICT:
      
      return self.predict(mode,table,scaffold,logits,dense_decoded)
      
    ctc_loss = self.ctc_loss(logits,labels,seqs_len)
    
    distillation_loss = self.distillation_loss(logits,teacher_logits,temperature = params.get('temperature'))
    
    loss = self.total_loss(ctc_loss, distillation_loss, alpha = params.get('alpha'), temperature = params.get('temperature'))

    if mode == tf.estimator.ModeKeys.TRAIN:
      
      return self.train(mode,loss,
                        quant_weights = quant_weights,
                        original_weights = original_weights,
                        lr = params.get('adam_lr'),
                        eps = params.get('adam_eps'),
                        clipping = params.get('clipping'),
                        bn = params.get('bn'))
    
    if mode == tf.estimator.ModeKeys.EVAL:
      
      return self.evaluate(mode,table,dense_decoded,labels,ctc_loss,scaffold)
  


