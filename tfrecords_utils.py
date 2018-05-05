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
from transcription_utils import create_vocab_id2transcript,get_ctc_char2ids,get_id2encoded_transcriptions,save_char_encoding
from audio_utils import get_audio
from collections import namedtuple

#######################################
# CTC LOSS : LARGEST VALUE 
# OF CHAR INIDICES MUST BE FOR `BLANK`
#######################################

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


class AudioExample(namedtuple('AudioExample', 'audio_path transcription')):
  
  def __str__(self):
    return '%s(%s, %s)' % (self.__class__.__name__, self.audio_path, self.transcription)
  

def parse_args():
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
  parser.add_argument('--limit', type = int, default = None, 
                      help = "Stop processing after having parsed this amount of audios. Default : stop only when job is done")
  
  return parser.parse_args()


def load_tfrecord_dataset(tfrecord_path, split, batch_size):
  
  dataset = tf.data.TFRecordDataset(tfrecord_path)
  
  dataset = dataset.map(parse_tfrecord_example, num_parallel_calls = 4)
  
  dataset = dataset.shuffle(buffer_size=2**16)
  
  if split == 'train':
    
    dataset = dataset.repeat()
    
  dataset = dataset.batch(batch_size)
  
  dataset = dataset.prefetch(batch_size)
    
  return dataset
  

def create_tfrecords_folder(out_path):

  out_path = Path(out_path)
    
  if not out_path.exists():
      
    out_path.mkdir()
      
    logger.info("Created folder `{}` where tfrecord files will be stored".format(str(out_path)))
        
  
def parse_tfrecord_example(proto):
  
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




def tfrecord_write_example(writer,audio, audio_shape, labels):
  
    example = tf.train.Example( features=tf.train.Features(
        feature={ 'audio': _float_feature(audio),
                 'labels': _int64_feature(labels),
                 'audio_shape' : _int64_feature(audio_shape)
                }))
    
    writer.write(example.SerializeToString())


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))  


def load_data_by_split(data_path, split, id2encoded_transc, limit):
  """
  Recursively load files from a LibriSpeech directory.
  """
    
  data = []
    
  parsed_file = 0
  
  main_path = Path(data_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir()]
  
  for split_folder in splits_folders:
    
    if split_folder.parts[-1].startswith(split):
      
      logger.info("Start processing folder `{}`".format(str(split_folder)))
      
      chapters = [chap for book in split_folder.iterdir() for chap in book.iterdir()]
      
      for chap in chapters:
        
        audio_files = [audio for audio in chap.glob('./*.flac')]
        
        for audio_file in audio_files:
          
          transcription_index = audio_file.parts[-1].strip(audio_file.suffix)
          
          labels = id2encoded_transc[transcription_index]
          
          data.append(AudioExample(str(audio_file),labels))
          
          parsed_file += 1
          
          if (parsed_file+1)%1000 == 0:
            
            logger.info("Loaded {} examples".format(parsed_file))
          
          if limit and parsed_file >= limit:
            
            logger.info("Successfully created {} audios examples".format(limit))
            
            break
          
          else:
            
            continue
          
        break
      
  return data


def write_tfrecords_by_split(out_path, split, data, sample_rate, form, **kwargs ):
  
  out_file = str(Path(out_path).joinpath('librispeech_tfrecords.' + split))
  
  logger.info("Examples will be stored in `{}`".format(str(out_file)))
  
  writer = tf.python_io.TFRecordWriter(out_file)
  
  for idx,audio_example in enumerate(data):
    
    audio = get_audio(audio_example.audio_path, sample_rate, form, **kwargs)
    
    labels = audio_example.transcription
    
    if form == 'raw':
      
      audio = audio[np.newaxis,:]
    
    audio_shape = list(audio.shape)
    print(audio_shape)
    
    audio = audio.flatten()
    
    tfrecord_write_example(writer = writer, audio =  audio, 
                                   audio_shape = audio_shape, labels = labels)
    
    if (idx+1)%1000 == 0:
      
      logger.info("Successfully wrote {} tfrecord examples".format(idx))
    
    
    
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
  
  
  
  
  
  
  
          
          
          
          
          
          
        
        
          
          
          
          
        
        
      
      
      
      
  
  
  
  
  
  
  
  
  
  
      
      
      
      
      
  
  
  
  





  
  
