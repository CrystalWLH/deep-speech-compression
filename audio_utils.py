#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:41:46 2018

@author: Samuele Garda
"""

import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def get_duration_in_s(audio, sample_rate):
  
  return len(audio) / sample_rate

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
  
  if form == 'ampl':
    
    audio = wave2ampl_spectrogram(audio,fft_window = kwargs['fft_window'],
                                  hop_length=kwargs['hop_length'])
    
  elif form == 'power':
    
    audio = wave2power_spectrogram(audio,fft_window = kwargs['fft_window'],
                                  hop_length=kwargs['hop_length'])
  elif form == 'mel':
    
    audio = wave2mel_spectrogram(audio, sample_rate, n_mels = kwargs['n_mels'])
                                 
  return audio