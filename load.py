#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:54:47 2018

@author: Samuele Garda
"""

import librosa
import numpy as np
from pathlib import Path

#TODO
# add mel/spectrogram

def get_transcriptions(trans_path : str ) -> dict:
  """
  Create transcriptions labels within folder
  """
  f = open(trans_path).readlines()
  
  trans_all = [trans.strip().split() for trans in f]
  
  id2trans = {trans[0] : trans[1:] for trans in trans_all}
  
  return id2trans
  


def load_raw_audion(audio_path : str, sample_rate : int) -> np.ndarry:
  """
  Load single audio file from path.
  """
  raw_audio = librosa.load(audio_path, sr = sample_rate) 
  
  return raw_audio

def raw2spectogram(audio : np.ndarray, fft_window : int, hop_length : int ):
  
  spectogram = librosa.stft(y=audio, n_fft= fft_window, hop_length= hop_length)

#def traverse_and_load(dir_path : str):
#  """
#  Recursively load files from a LibriSpeech directory.
#  """
#  
#  main_path = Path(dir_path)
#  
#  audio_files = main_path.glob('**/*.flac')
#  
#  
#      
#      
      
      
      
  
  
  
  





  
  