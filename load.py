#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:54:47 2018

@author: Samuele Garda
"""

import os
import tensorflow as tf
import pickle
import librosa
import numpy as np
from pathlib import Path

#TODO
# add mel/spectrogram

######################
# CTC LOSS : LARGEST VALUE OF CHAR INIDICES MUST BE FOR `BLANK`
######################


SAMPLE_RATE = 16000


def save_char_encoding(char_enc, path):
  
  pickle.dumb(open(path + '.vocab', 'wb'), char_enc)


def sent_char_to_ids(sent,mapping):
  
  return [mapping.get(c) for c in sent]


def get_duration_in_s(audio, sample_rate):
  
  return len(audio) / sample_rate


def get_ctc_char2ids(chars_set):
  
  chars_set.remove(' ')
  char2id = {c : idx for idx,c in chars_set}
  char2id[' '] = len(char2id)
  
  return char2id

def get_id2encoded_transcriptions(ids2trans,mapping):
  
  ids2encoded_trans = {ref : sent_char_to_ids(sent,mapping) for ref,sent in ids2trans.items()}
  
  return ids2encoded_trans
  

def create_vocab_id2transcript(dir_path, char2ind_path):
  """
  Create transcriptions labels within folder
  """
  
  chars_set = set()
  
  ids2transcriptions = {}
  
  main_path = Path(dir_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir()]
  
  trans_files = [trans for book in splits_folders for trans in book.glob('**/*.txt')]
  
  for trans_file in trans_files:
    
    with trans_file.open() as tr_file:
      for line in tr_file:
        split_line = line.strip().split()
        
        ref,sent_chars = split_line[0], list(' '.join(split_line[1:]))
        
        ids2transcriptions[ref] = sent_chars
        
        chars_set.update(sent_chars)
        
        
  return chars_set, ids2transcriptions
  
  

def load_raw_audio(audio_path , sample_rate):
  """
  Load single audio file from path.
  """
  raw_audio = librosa.load(audio_path, sr = sample_rate)[0]
  
  return raw_audio

def wave2ampl_spectrogram(audio, fft_window, hop_length):
  
  spectogram = librosa.stft(y=audio, n_fft= fft_window, hop_length= hop_length)
  
  return np.abs(spectogram)

def wave2power_spectrogram(audio, fft_window, hop_length):
  
  spectogram = librosa.stft(y=audio, n_fft= fft_window, hop_length= hop_length)
  
  return np.abs(spectogram) ** 2

def wave2mel_spectrogram(audio, sample_rate, n_mels):
  
  S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels)
  log_S = librosa.power_to_db(S, ref=np.max)
  
  return log_S


def get_audio(audio_path, sample_rate, form, **kwargs):
  
  audio = load_raw_audio(audio_path, sample_rate)
  
  if form == 'ampl_spec':
    
    audio = wave2ampl_spectrogram(audio, **kwargs)
    
  elif form == 'power_spec':
    
    audio = wave2power_spectrogram(audio,**kwargs)
    
  elif form == 'mel_spec':
    
    audio = wave2mel_spectrogram(audio, sample_rate, **kwargs)
    
  return audio

def traverse_and_load(dir_path, split, id2encoded_transc, sample_rate, form, **kwargs ):
  """
  Recursively load files from a LibriSpeech directory.
  """
  
  main_path = Path(dir_path)
  
  splits_folders = [child for child in main_path.iterdir() if child.is_dir()]
  
  for split_folder in splits_folders:
    
    if split_folder.parts[-1].startswith(split):
      
      chapters = [chap for book in split_folder.iterdir() for chap in book.iterdir()]
      
      for chap in chapters:
        
        audio_files = [audio for audio in chap.glob('./*.flac')]
        
        for audio_file in audio_files:
          
          audio = get_audio(str(audio_file), sample_rate, form, **kwargs)
          
          transcription_index = audio_file.parts[-1].strip(audio_file.suffix)
          
          labels = id2encoded_transc[transcription_index]
          
          print(audio)
          
          print(labels)
          
          
          
          
          
          
        
        
          
          
          
          
        
        
      
      
      
      
  
  
  
  
  
  
  
  
  
  
      
      
      
      
      
  
  
  
  





  
  