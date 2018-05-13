#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:54:47 2018

@author: Samuele Garda
"""

import argparse
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils.transcription_utils import create_vocab_id2transcript,get_ctc_char2ids,get_id2encoded_transcriptions,save_char_encoding
from utils.audio_utils import get_audio
from collections import namedtuple

#######################################
# CTC LOSS : LARGEST VALUE 
# OF CHAR INIDICES MUST BE FOR `BLANK`
#######################################

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_args():
  """
  Parse arguments for script.
  """
  parser = argparse.ArgumentParser(description='Create tfrecord files from LibriSpeech corpus')
  parser.add_argument('-d', '--data', required=True, type = str, help='Path to unzipped LibriSpeech dataset')
  parser.add_argument('-s', '--split', required=True, type = str, 
                      choices = ('train','dev','test'), help='Which dataset split to be parsed')
  parser.add_argument('-o', '--out', required=True, type = str, help='Directory where to store tfrecord files')
  parser.add_argument('-f', '--format', type = str, default = 'ampl', choices = ('raw','ampl','power','mel'),
                      help='Representation of the audio to be used: `raw` (wave), `ampl` : amplitude spectrogram,\
                      `power` : power spectrogram, `mel` : mel spectrogram. Default to `ampl`')
  parser.add_argument('-l', '--loss' , type = str, default = 'ctc', choices = ('ctc','asg'), 
                      help = 'Specify loss function since affects encoding of characters in transcriptions')
  parser.add_argument('--sr', type = int, default = 16000, help = 'Sample rate with which audios are loaded. Default to : 16000')
  parser.add_argument('--limit',type = int, default = None, 
                      help = "Stop processing after having parsed this amount of audios. Default : stop only when job is done")
  
  return parser.parse_args()


class AudioExample(namedtuple('AudioExample', 'audio_path transcription')):
  """
  Namedtuple custom class. It stores `audio_path` a string representing path to the audio and `transcription`, i.e. 
  the correspondent encoded (int) transcription.
  """
  
  def __str__(self):
    return '%s(%s, %s)' % (self.__class__.__name__, self.audio_path, self.transcription)
  

def create_tfrecords_folder(out_path):
  """
  Create folder where tfrecord files will be stored
  
  :param:
    out_path (str) : folder where to store tfrecords files
    
  :return:
    out_path (pathlib.Path) : Path object
  """
  

  out_path = Path(out_path)
    
  if not out_path.exists():
      
    out_path.mkdir()
      
    logger.info("Created folder `{}` where tfrecord files will be stored".format(str(out_path)))
    
  return out_path
    

def tfrecord_write_example(writer,audio, audio_shape,seq_len, labels):
  """
  Write example to TFRecordWriter.
  
  :param:
    writer (tf.python_io.TFRecordWriter) : file where examples will be stored
    audio (np.ndarry) : audio
    audio_shape (list) : shape of audio representation
    labels (list) : encoded transcription
  """
  
  example = tf.train.Example( features=tf.train.Features(
      feature={'audio': _float_feature(audio),
               'audio_shape' : _int64_feature(audio_shape),
               'seq_len' : _int64_feature([seq_len]),
               'labels': _int64_feature(labels),
              }))
  
  writer.write(example.SerializeToString())


def _int64_feature(value):
  """
  Map list of ints to tf compatible Features
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """
  Map list of floats to tf compatible Features
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))  


def load_data_by_split(data_path, split, id2encoded_transc, limit):
  """
  Recursively load files from a LibriSpeech directory. If limit is not specified stop only when job is done.
  
  :param:
    data_path (str) : path to main folder of LibriSpeech corpus
    split (str) : part of the dataset to load. Choiches = (`train`,`dev`,`test`)
    id2encoded_transc (dict) : dictionary of transcription ids (keys) and the transcription (values) in list of ints format
    limit (int) : when to stop
  :return:
    data (list) : list of AudioExample objects
  """
    
  data = []
      
  main_path = Path(data_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir()]
  
  def audio_paths_gen():
    """
    Generator of audio paths. Avoid loading all of them into memory
    """
    for split_folder in splits_folders:
      for audio in split_folder.glob('**/*.flac'):
        yield audio
  
  for idx,audio_file in enumerate(audio_paths_gen()):
    
    transcription_index = audio_file.parts[-1].strip(audio_file.suffix)
          
    labels = id2encoded_transc[transcription_index]
          
    data.append(AudioExample(str(audio_file),labels))
      
    if limit and idx >= limit:
      
      logger.info("Successfully loaded {} audios examples".format(limit))
      
      break
  
  return data


def write_tfrecords_by_split(out_path, split, data, sample_rate, form, **kwargs ):
  """
  Write data loaded with `load_data_by_split` to a tf record file. If form is not specified raw audio are loaded. Otherwise `kwargs` passed
  to transformation function.
  Raw audio expanded to 2D for compatibility with input to Convolution operations.
  
  :param:
    out_path (str) : folder where to store tfrecord data
    split (str) : part of dataset
    data (list) : list of AudioExample objects
    sample_rate (int) : rate at which audio was sampled when loading
    form (str) : representation. Choices = (`ampl`,`power`,`mel`)
    
  """
  
  out_path = create_tfrecords_folder(out_path)
  
  out_file = str(out_path.joinpath('librispeech_tfrecords.' + split))
  
  logger.info("Examples will be stored in `{}`".format(str(out_file)))
  
  writer = tf.python_io.TFRecordWriter(out_file)
  
  for idx,audio_example in enumerate(data,start = 1):
    
    audio = get_audio(audio_example.audio_path, sample_rate, form, **kwargs)
    
    labels = audio_example.transcription
    
    if form == 'raw':
      
      audio = audio[np.newaxis,:]
    
    audio_shape = list(audio.shape)
    
    seq_len = audio_shape[1]
        
    audio = audio.flatten()
    
    tfrecord_write_example(writer = writer, audio =  audio, 
                                   audio_shape = audio_shape,seq_len = seq_len,labels = labels)
    
    if (idx)%1000 == 0:
      
      logger.info("Successfully wrote {} tfrecord examples".format(idx))
      
  writer.close()   
      
  logger.info("Completed writing examples in tfrecords format at `{}`".format(out_file))
    
    
    
if __name__ == "__main__":
  
  args = parse_args()
  
  logger.info("Start process for creating tfrecord files for split : `{}`".format(args.split))
  
  create_tfrecords_folder(args.out)
  
  chars_set, ids2trans = create_vocab_id2transcript(args.data)
  
  if args.loss == 'ctc':
    
    chars2ids = get_ctc_char2ids(chars_set)
    
  elif args.loss == 'asg':
    
    raise NotImplementedError("Sorry! ASG loss is not available!")
    
  save_char_encoding(chars2ids, args.out)
    
  encoded_transcriptions = get_id2encoded_transcriptions(ids2trans, chars2ids)
  
  split_data = load_data_by_split(data_path = args.data, split = args.split,
                                 id2encoded_transc= encoded_transcriptions, limit = args.limit )
  
  write_tfrecords_by_split(data= split_data, out_path = args.out, split = args.split,
                               sample_rate = args.sr, form = args.format,
                               fft_window = 512, hop_length = 128, n_mels = 128)
  
  
  
  
  
  
  
          
          
          
          
          
          
        
        
          
          
          
          
        
        
      
      
      
      
  
  
  
  
  
  
  
  
  
  
      
      
      
      
      
  
  
  
  





  
  
