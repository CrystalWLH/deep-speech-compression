#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:54:47 2018

@author: Samuele Garda
"""

import argparse
import logging
import tensorflow as tf
from pathlib import Path
from transcription_utils import create_vocab_id2transcript,get_ctc_char2ids,get_id2encoded_transcriptions
from audio_utils import get_audio

#######################################
# CTC LOSS : LARGEST VALUE 
# OF CHAR INIDICES MUST BE FOR `BLANK`
#######################################

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


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
  


def create_tfrecords_folder(out_path):

  out_path = Path(out_path)
    
  if not out_path.exists():
      
    out_path.mkdir()
      
    logger.info("Created folder `{}` where tfrecord files will be stored".format(str(out_path)))
        
  
def parse_tfrecord_example(proto):
  
    features = {"audio": tf.VarLenFeature(tf.float32),
                "shape": tf.FixedLenFeature((2,), tf.int64),
                "labels": tf.VarLenFeature(tf.int64)}
    
    parsed_features = tf.parse_single_example(proto, features)
    sparse_audio = parsed_features["audio"]
    shape = tf.cast(parsed_features["audio_shape"], tf.int32)
    dense_audio = tf.reshape(tf.sparse_to_dense(sparse_audio.indices,
                                                sparse_audio.dense_shape,
                                                sparse_audio.values),shape)
    
    sparse_trans = parsed_features["transcription"]
    dense_trans = tf.sparse_to_dense(sparse_trans.indices,
                                     sparse_trans.dense_shape,
                                     sparse_trans.values)
    return dense_audio, tf.cast(dense_trans, tf.int32)




def tfrecord_write_example(writer,audio,shape, labels):
  
    example = tf.train.Example( features=tf.train.Features(
        feature={ 'audio': _float_feature(audio),
                 'labels': _int64_feature(labels),
                 'shape' : _int64_feature(shape)
                }))
    
    writer.write(example.SerializeToString())


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))  


def load_data_or_write_tfrecords(data_path, out_path, split, id2encoded_transc, mode = 'tfrecord',
                      sample_rate = 16000, form = 'raw', limit = None, **kwargs ):
  """
  Recursively load files from a LibriSpeech directory.
  """
  
  if mode == 'tfrecord':
      
    out_file = str(Path(out_path).join('librispeech_tfrecords.' + split))
    
    logger.info("Examples witll be stored in `{}`".format(str(out_file)))
    
    writer = tf.python_io.TFRecordWriter(out_file)
    
  elif mode == 'load':
    
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
          
          audio = get_audio(str(audio_file), sample_rate, form, **kwargs).flatten()
          
          transcription_index = audio_file.parts[-1].strip(audio_file.suffix)
          
          labels = id2encoded_transc[transcription_index]
          
          audio_shape = list(audio.shape)
          
          if mode == 'load':
            
            data.append((audio,labels))
            
          elif mode == 'tfrecord':
          
            tfrecord_write_example(writer, audio, audio_shape, labels)
          
          parsed_file += 1
          
          if (parsed_file+1)%1000 == 0:
            
            logger.info("{} examples parsed to `{}`".format(parsed_file,out_file))
            
          
            
          if limit and parsed_file >= limit:
            
            logger.info("Successfully loaded {} audios examples".format(limit))
            
            break
          
    if mode == 'load':
      
      return data
          
    
    
if __name__ == "__main__":
  
  args = parse_args()
  
  logger.info("Start process for creating tfrecord files for split : `{}`".format(args.split))
  
  create_tfrecords_folder(args.out)
  
  chars_set, ids2trans = create_vocab_id2transcript(args.data)
  
  if args.loss == 'ctc':
    
    chars2ids = get_ctc_char2ids(chars_set)
    
  elif args.loss == 'asg':
    
    raise NotImplementedError("Sorry! ASG loss is not available!")
    
  encoded_transcriptions = get_id2encoded_transcriptions(ids2trans, chars2ids)
  
  create_tfrecords_folder(args.out)
  
  load_data_or_write_tfrecords(data_path= args.data, out_path = args.out, split = args.split,
                               id2encoded_transc = encoded_transcriptions, mode = 'tfrecord',
                               sample_rate = args.sr, form = args.format, limit = args.limit,
                               fft_window = 512, hop_length = 128, n_mels = 128)
  
  
  
  
  
  
  
          
          
          
          
          
          
        
        
          
          
          
          
        
        
      
      
      
      
  
  
  
  
  
  
  
  
  
  
      
      
      
      
      
  
  
  
  





  
  