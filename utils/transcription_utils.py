#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:40:27 2018

@author: Samuele Garda
"""

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def decoder_dict(char2idx):
  """
  Create lookup index to character for decoding predictions.
  
  :param:
    char2idx (dict) : lookup char-idx
  :return:
    inverse of `char2idx`
  """
  
  return {idx : char for char,idx in char2idx.items()}


def decode_sequence(seq, idx2char):
  """
  Decode sequence of indices to sequence of characters.
  
  :param:
    seq (list) : list of ints indices
    idx2char (dict) : idx-char lookup
  """
  
  return [idx2char.get(c) for c in seq]

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
  
    logger.info("Saved character to indices lookup at `{}`".format(path))


def sent_char_to_ids(sent,mapping):
  """
  Map sentece to ids. 
  
  Input MUST BE LIST OF CHARACTERS.
  
  :param:
    sent (list) : list of characters
    mapping (dict) : characters lookup (char2id)
    
  :return:
    mappend sentence (list)
  """
  
  return [mapping.get(c) for c in sent]


def get_ctc_char2ids(chars_set):
  """
  Transform character lookup for CTC loss. 
  Blank char must be the last (highest lookup ID). See https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss 
  Enumeration starts from `1`. Id `0` kept for padding. 
  
  :param:
    char_set (set) : set of characters
  :return:
    char2id (dict) : character lookup
  """
  
  char2id = {c : idx for idx,c in enumerate(chars_set)}
  char2id['blank'] = len(char2id)
  
  logger.info("Modified characters id lookup for compatibility with CTC loss")
  
  return char2id

def get_id2encoded_transcriptions(ids2trans,mapping):
  """
  Create lookup table for transcriptions. Each transcription id is associeted with its encoded representation (char2id).
  
  :param:
    id2trans (dict) : dictionary of transcription ids (keys) and the actual transcription (values) in list of chars format
    mapping (dict) : characters lookup
  :return:
    ids2encoded_trans (dict) :  dictionary of transcription ids (keys) and the transcription (values) in list of ints format
  
  """
  
  ids2encoded_trans = {ref : sent_char_to_ids(sent,mapping) for ref,sent in ids2trans.items()}
  
  logger.info("Encoding transcription with character to indices lookup")
  
  return ids2encoded_trans
  

def create_vocab_id2transcript(dir_path):
  """
  Traverse LibriSpeech corpus and create:
    - lookup table for transcriptions 
    - vocabulary : character set
    
  :param:
    dir_path (str) : path to main folder of LibriSpeech corpus
  :return:
    char_set (set) : character set
    ids2transcriptions (dict) :  dictionary of transcription ids (keys) and the actual transcription (values) in list of chars format
  """
  
  chars_set = set()
  
  ids2transcriptions = {}
  
  main_path = Path(dir_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir()]
  
  def trans_files_gen():
    """
    Generator of transcription file paths. Avoid loading all of them into memory
    """
    for split_folder in splits_folders:
      for trans in split_folder.glob('**/*.txt'):
        yield trans
        
  logger.info("Created transcriptions file generator")
  
  for idx,trans_file in enumerate(trans_files_gen(),start = 1):
    
    if (idx%1000) == 0:
      
      logger.info("Processing {}th transcription file".format(idx))
    
    with trans_file.open() as tr_file:
      for line in tr_file:
        split_line = line.strip().split()
        
        ref,sent_chars = split_line[0], list(' '.join(split_line[1:]))
        
        ids2transcriptions[ref] = sent_chars
        
        chars_set.update(sent_chars)
        
        
        
  logger.info("Created character set of size {}".format(len(chars_set)))
  logger.info("Created transcriptions lookup")
  
  return chars_set, ids2transcriptions


