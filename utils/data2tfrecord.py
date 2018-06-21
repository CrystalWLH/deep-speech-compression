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
import pickle
from utils.transcription import create_id2transcript,get_ctc_char2ids,get_id2encoded_transcriptions,chars2ids_to_file
from utils.audio import partial_get_audio_func,get_audio_id
import multiprocessing as mp
import time
from functools import partial

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
  parser.add_argument('--cached-ids2trans', type = str,default = None, help='Path to audio_id-transcription lookup. Do not call this argument if you wish to create this lookup')
  parser.add_argument('--cores', type = int,default = 2, help='Number of cores to use. Default : 2')
  parser.add_argument('--chunck-size', type = int,default = 100, help='How many audio examples to send to worker to be processed. Default : 100')
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



def load_pickle(file_path):
  """
  Util funciton to load character lookup from pickle format 
  
  :param:
    char_enc (dict) : character lookup
    path (str) : path where to store item
    
  :return:
    item : whathever was saved in the file
  """
  path = Path(file_path)
  
  if path.exists():
    
    item = pickle.load(open(str(path), mode = "rb"))
  
    return item
  
  else:
    raise ValueError("File {} not found!".format(file_path))


def save_pickle(char_enc, path, file_name):
  """
  Util funciton to save in pickle format character lookup.
  
  :param:
    char_enc (dict) : character lookup
    path (str) : path where to store item
  """
  path = Path(path).joinpath(file_name)
  
  if not path.exists():
  
    pickle.dump( char_enc, open( str(path), 'wb'))
  
    logger.info("Saved item at `{}`".format(path))


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
    data (list) : list of dictionaries
  """
    
  data = []
      
  audio_paths = get_audio_files(data_path)
  
  for idx,audio_file in enumerate(audio_paths,start = 1):
    
    audio_id = get_audio_id(audio_file)
          
    labels = id2encoded_transc[audio_id]
          
    data.append({'audio' : str(audio_file), 'labels' : labels})
      
    if limit and idx >= limit:
      
      logger.info("Successfully loaded {} audios examples".format(limit))
      
      break
  
  if not limit:  
    logger.info("Successfully loaded {} audios examples".format(len(audio_paths)))
  
  return data

def transform_sample(audio_path,preprocess_func):
  """
  Transform audio with function.
  
  :param:
    audio_path (str) : path to audio
    preprocess_func (function) : transformation function
  :return:
    audio_id (str) : id of audio
    audio_data (np.ndarray) : audio data
    
  """
  audio_data = preprocess_func(audio_path)
  audio_id = get_audio_id(audio_path)
  return audio_id,audio_data
  

def get_audio_files(data_path):
  """
  Create generator of audio paths.
  
  :param:
    data_path (str) : path to main folder of LibriSpeech corpus
  """
  
  main_path = Path(data_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir() and split in str(child)]
  
  for split_folder in splits_folders:
    for audio in split_folder.glob('**/*.flac'):
      yield str(audio)



def write_tfrecords_by_split(data_path,out_path,split,id2encoded_transc, sample_rate, form, n_fft, hop_length, n_mfcc, chunck_size,cores):
  """
  Write data loaded with `load_data_by_split` to a tf record file. If form is not specified raw audio are loaded. 
  Raw audio expanded to 2D for compatibility with input to Convolution operations.
  
  :param:
    data_path (str) : path to main folder of LibriSpeech corpus
    out_path (str) : folder where to store tfrecord data
    id2encoded_transc (dict) : dictionary of transcription ids (keys) and the transcription (values) in list of ints format
    split (str) : part of dataset
    data (list) : list of AudioExample objects
    sample_rate (int) : rate at which audio was sampled when loading
    form (str) : representation. Choices = (`raw`,`power`,`mfccs`)
    n_mfcc (int) : the number of coefficients for mfccs 
    n_fft (int) : the window size of the fft
    hop_length (int): the hop length for the window
    chunck_size (int) : how many batched items to send to worker
    cores (int) :Number of cores to use. Default : 2
    
  """
    
  out_path = create_tfrecords_folder(out_path)
  
  out_file = str(out_path.joinpath('tfrecords_{}.{}'.format(form,split.strip('-'))))
    
  logger.info("Examples will be stored in `{}`".format(str(out_file)))
  
  writer = tf.python_io.TFRecordWriter(out_file)
  
  get_audio = partial_get_audio_func(form, sample_rate, n_fft, hop_length, n_mfcc)
  
  transform_sample_part = partial(transform_sample,preprocess_func = get_audio )
  
  logger.info("Start processing audio files")
  
  pool = mp.Pool(cores)
  
  start_time = time.time()
  
  for idx,(audio_id,audio) in enumerate(pool.imap_unordered(transform_sample_part,get_audio_files(data_path),chunck_size), start = 1):
        
    labels = id2encoded_transc.get(audio_id)
        
    audio_shape = list(audio.shape)
    
    if idx == 1:
      logger.info("Number of input channel is {}".format(audio_shape[0]))
  
    audio = audio.flatten()
      
    tfrecord_write_example(writer = writer, audio =  audio, 
                                     audio_shape = audio_shape,labels = labels)
    if (idx)%1000 == 0:
      
      end = time.time()
        
      logger.info("Successfully parsed and saved to tfrecord {} examples in {}".format(idx,end - start_time))
      
      start_time = time.time()
        
  writer.close()   
      
  logger.info("Completed writing examples in tfrecords format at `{}`".format(out_file))
  
if __name__ == "__main__":
    
  args = parse_args()
    
  create_tfrecords_folder(args.out)
  
  if args.loss == 'ctc':
  
    if not args.cached_ids2trans:
  
      ids2trans = create_id2transcript(args.data)
        
      save_pickle(ids2trans, args.out, 'ids2transc.pkl')
      
      chars2ids = get_ctc_char2ids()
            
      chars2ids_to_file(chars2ids, args.out, 'ctc_vocab.txt')
    
    else:
    
      logger.info("Loading cached id-transcriptions lookup")
      
      ids2trans = load_pickle(args.cached_ids2trans)
      
      chars2ids = get_ctc_char2ids()
    
       
  elif args.loss == 'asg':
      
    raise NotImplementedError("Sorry! ASG loss is not available!")
    
    
  encoded_transcriptions = get_id2encoded_transcriptions(ids2trans, chars2ids)
  
  for split in args.splits.split(','):
    
    logger.info("\n\nProcessing files in `{}`\n".format(split))
  
    write_tfrecords_by_split(data_path = args.data,
                             out_path = args.out,
                             split = split,
                             id2encoded_transc = encoded_transcriptions,
                             sample_rate = args.sr,
                             form = args.format,
                             n_fft = 512,
                             hop_length = 160,
                             n_mfcc = 13,
                             chunck_size = args.chunck_size,
                             cores = args.cores)
  
  
 