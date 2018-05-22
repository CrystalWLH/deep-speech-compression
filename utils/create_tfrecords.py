#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:54:47 2018

@author: Samuele Garda
"""

import argparse
import logging
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils.transcription_utils import create_vocab_id2transcript,get_ctc_char2ids,get_id2encoded_transcriptions,save_pickle
from utils.audio_utils import get_audio,normalize
from collections import namedtuple
import multiprocessing as mp
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
  parser.add_argument('-s', '--splits', required=True, type = str,
                      help="Comma separated list of either folders (`dev-others`), split (train) or mixed. \
                      If generic split is defined (e.g. `train`) folder for that split will be merged into single file.")
  parser.add_argument('-o', '--out', required=True, type = str, help='Directory where to store tfrecord files')
  parser.add_argument('-f', '--format', type = str, default = 'mfccs', choices = ('raw','power','mfccs'),
                      help='Representation of the audio to be used: `raw` (wave),`power` : power spectrogram, `mfccs` : MFCCs . Default to `mfccs`')
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
    

def tfrecord_write_example(writer,audio, audio_shape, labels):
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
    split (str) : part of the dataset to load. Either a split (`train`,`dev`,`test`) or a specific folder e.g. (`dev-other`) 
    id2encoded_transc (dict) : dictionary of transcription ids (keys) and the transcription (values) in list of ints format
    limit (int) : when to stop
  :return:
    data (list) : list of AudioExample objects
  """
    
  data = []
      
  main_path = Path(data_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir() and split in str(child)]
  
  logger.info("Processing : {}".format([str(s) for s in splits_folders]))
  
  audio_paths = [audio for split_folder in splits_folders for audio in split_folder.glob('**/*.flac')]
  
  logger.info("Loaded {} audio paths".format(len(audio_paths)))
  
  if split.startswith('train') or split.startswith('dev'):
    
    logger.info("Shuffling data before parsing")
    
    random.shuffle(audio_paths)
  
  for idx,audio_file in enumerate(audio_paths,start = 1):
    
    transcription_index = audio_file.parts[-1].strip(audio_file.suffix)
          
    labels = id2encoded_transc[transcription_index]
          
    data.append(AudioExample(str(audio_file),labels))
      
    if limit and idx >= limit:
      
      logger.info("Successfully loaded {} audios examples".format(limit))
      
      break
  
  if not limit:  
    logger.info("Successfully loaded {} audios examples".format(len(audio_paths)))
  
  return data

def _preprocessing_error_callback(error):
    raise RuntimeError('An error occurred during preprocessing') from error



def write_tfrecords_by_split(out_path, split, data, sample_rate, form, n_fft, hop_length, n_mfcc):
  """
  Write data loaded with `load_data_by_split` to a tf record file. If form is not specified raw audio are loaded. 
  Raw audio expanded to 2D for compatibility with input to Convolution operations.
  
  :param:
    out_path (str) : folder where to store tfrecord data
    split (str) : part of dataset
    data (list) : list of AudioExample objects
    sample_rate (int) : rate at which audio was sampled when loading
    form (str) : representation. Choices = (`raw`,`power`,`mfccs`)
    n_mfcc (int) : the number of coefficients for mfccs 
    n_fft (int) : the window size of the fft
    hop_length (int): the hop length for the window
    
  """
  
  out_path = create_tfrecords_folder(out_path)
  
  out_file = str(out_path.joinpath('tfrecords_{}.{}'.format(form,split.strip('-'))))
  
  logger.info("Examples will be stored in `{}`".format(str(out_file)))
  
  writer = tf.python_io.TFRecordWriter(out_file)
  
  pool = mp.Pool(processes= mp.cpu_count())

  arguments_to_map = [(audio_example.audio_path, sample_rate, form, n_fft, hop_length, n_mfcc) for audio_example in data]
  
  labels = [audio_example.transcription for audio_example in data]
  
  logger.info("Computing audio features representation")
      
  audios = pool.starmap_async(get_audio, arguments_to_map, error_callback= _preprocessing_error_callback).get() 
  
  logger.info("Finished computing audio feature representation")
  
  for idx,(audio,label) in enumerate(zip(audios,labels),start = 1):
    
    if form == 'raw':
      audio = normalize(audio[np.newaxis, :])
      
    audio_shape = list(audio.shape)
    
    if idx == 1:
    
      logger.info("Number of input channels is : {}".format(audio_shape[0]))
                    
    audio = audio.flatten()
      
    tfrecord_write_example(writer = writer, audio =  audio, 
                                     audio_shape = audio_shape,labels = label)
    if (idx)%1000 == 0:
        
      logger.info("Successfully wrote {} tfrecord examples".format(idx))
  

  writer.close()   
      
  logger.info("Completed writing examples in tfrecords format at `{}`".format(out_file))
  
if __name__ == "__main__":
    
  args = parse_args()
    
  create_tfrecords_folder(args.out)
  
  chars_set, ids2trans = create_vocab_id2transcript(args.data)
  
  chars = [c for ids,trans in ids2trans.items() for c in trans]
    
  if args.loss == 'ctc':
    
    chars2ids = get_ctc_char2ids(chars_set)
    
  elif args.loss == 'asg':
    
    raise NotImplementedError("Sorry! ASG loss is not available!")
    
  save_pickle(chars2ids, args.out, 'vocab.pkl')
      
  encoded_transcriptions = get_id2encoded_transcriptions(ids2trans, chars2ids)
  
  for split in args.splits.split(','):
    
    logger.info("\n\nProcessing files in `{}`\n\n".format(split))
  
    split_data = load_data_by_split(data_path = args.data, split = split,
                                   id2encoded_transc= encoded_transcriptions, limit = args.limit )
    
    write_tfrecords_by_split(data= split_data, out_path = args.out, split = split,
                                 sample_rate = args.sr, form = args.format,
                                 n_fft = 400, hop_length = 160, n_mfcc = 40)
  
  
  
  
  
  
  
          
          
          
          
          
          
        
        
          
          
          
          
        
        
      
      
      
      
  
  
  
  
  
  
  
  
  
  
      
      
      
      
      
  
  
  
  





  
  
