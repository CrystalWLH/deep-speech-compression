#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:41:46 2018

@author: Samuele Garda
"""

import os
import logging
import numpy as np
import librosa
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def get_audio_id(audio_path):
  """
  From audio path return audio id
  
  :param:
    audio_path (str) : path to audio
  :return:
    audio_id (str) : id of audio
  """
  
  file_name = os.path.basename(audio_path)
  audio_id = os.path.splitext(file_name)[0]
  
  return audio_id


def normalize(audio):
  """
  Normalize audio signal to Mean 0 std 1
  
  :param:
    audio (np.ndarray) : audio examples
    
  :return:
    
    normalized audio
  """
  
  return (audio - np.mean(audio)) / np.std(audio)

def get_duration_in_s(audio, sample_rate):
  """
  Compute audio example duration in seconds.
  
  :param:
    audio (np.ndarray) : audio examples (wave)
    sample_rate (int) : rate at which audio was sampled when loading
    
  :return:
    Audio length in seconds
  """
  
  return len(audio) / sample_rate

def load_wave(audio_path , sample_rate = 16000):
  """
  Load single audio file from path.
  
  :param:
    audio_path (str) : path to audio
    sample_rate (int) : rate at which audio is sampled
  :return:
    raw_audio (np.ndarray) : audio examples (wave)
  """
  
  audio,sr = librosa.load(path = audio_path, sr = sample_rate)
  
  return normalize(audio[np.newaxis,:])
  
  
def load_mfccs(audio_path, sample_rate = 16000, n_mfcc=13, n_fft=512, hop_length=160):
  """
  Load audio and calculate mfcc coefficients from the given raw audio data
  params:
    audio_path (str) : path to audio
    sample_rate (int) : the sample rate of the audio
    n_mfcc (int) : the number of coefficients to generate
    n_fft (int) : the window size of the fft
    hop_length (int): the hop length for the window
  return:
    mfcc (np.ndarray) : the mfcc coefficients in the form [coefficients, time ]
  """
  
  audio,sr = librosa.load(path = audio_path, sr = sample_rate)
  
  mfcc = librosa.feature.mfcc(y = audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

  # add derivatives and normalize
  mfcc_delta = librosa.feature.delta(mfcc)
  mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
  mfcc = np.concatenate((normalize(mfcc),
                         normalize(mfcc_delta),
                         normalize(mfcc_delta2)), axis=0)
  
  
  return mfcc

def load_pow_spec(audio_path, sample_rate, n_fft=512, hop_length=160):
  """
  Load audio and compute power spectrogram from wave audio.
  
  :param:
    audio_path (str) : path to audio
    sample_rate (int) : the sample rate of the audio
    fft_window (int) : window for FFT
    hop_length (int) : number audio of frames between FFT columns
  :return:
    spectrogram (np.ndarray) : Power spectrogram
    
  """
  
  audio,sr = librosa.load(path = audio_path, sr = sample_rate)
  
  spectogram = librosa.stft(y=audio, n_fft= n_fft, hop_length= hop_length)
  
  return normalize(np.abs(spectogram) ** 2)


def partial_get_audio_func(form, sample_rate, n_fft, hop_length, n_mfcc):
  """
  Create loading function with single parameter, i.e. path to audio. 
  
  :param:
    form (str) : representation. Choices = (`raw`,`power`,`mfccs`)
    sample_rate (int) : the sample rate of the audio
    n_mfcc (int) : the number of coefficients to generate
    n_fft (int) : the window size of the fft
    hop_length (int): the hop length for the window
  :return:
    func (function) : function with fixed parameters
    
  """
  
  if form == 'raw':
    
    func = partial(load_wave, sample_rate = sample_rate)
    
  elif form == 'power':
    
    func = partial(load_pow_spec, sample_rate = sample_rate, n_fft = n_fft, hop_length = hop_length)
    
  elif form == 'mfccs':
    
    func = partial(load_mfccs, sample_rate = sample_rate, n_fft = n_fft, hop_length = hop_length, n_mfcc = n_mfcc)
    
  return func
  
