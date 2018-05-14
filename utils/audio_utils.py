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


def z_normalization(audio):
  """
  Normalize audio signal. Mean 0 std 1
  
  :param:
    audio (np.ndarray) : audio examples
    
  :return:
    
    normalized audio
  """
  
  
  return np.mean(audio) / np.std(audio)

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

def load_raw_audio(audio_path , sample_rate):
  """
  Load single audio file from path.
  
  :param:
    audio_path (str) : path to audio
    sample_rate (int) : rate at which audio is sampled
  :return:
    raw_audio (np.ndarray) : audio examples (wave)
  """
  raw_audio = librosa.load(audio_path, sr = sample_rate)[0]
  
  return raw_audio

def wave2ampl_spectrogram(audio, fft_window, hop_length):
  """
  Compute amplitude spectrogram from wave audio.
  
  :param:
    audio (np.ndarray) : audio examples (wave)
    fft_window (int) : window for FFT
    hop_length (int) : number audio of frames between FFT columns
  :return:
    spectrogram (np.ndarray) : amplitude spectrogram
  """
  
  spectogram = librosa.stft(y=audio, n_fft= fft_window, hop_length= hop_length)
  
  return np.abs(spectogram)

def wave2power_spectrogram(audio, fft_window, hop_length):
  """
  Compute power spectrogram from wave audio.
  
  :param:
    audio (np.ndarray) : audio examples (wave)
    fft_window (int) : window for FFT
    hop_length (int) : number audio of frames between FFT columns
  :return:
    spectrogram (np.ndarray) : Power spectrogram
  """
  
  spectogram = librosa.stft(y=audio, n_fft= fft_window, hop_length= hop_length)
  
  return np.abs(spectogram) ** 2

def wave2mel_spectrogram(audio, sample_rate, n_mels):
  """
  Compute power spectrogram from wave audio.
  
  :param:
    audio (np.ndarray) : audio examples (wave)
    n_mels (int) : n mel frequencies
  :return:
    log_S (np.ndaray) : mel  spectrogram
  """
  
  
  S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels)
  log_S = librosa.power_to_db(S, ref=np.max)
  
  return log_S


def get_audio(audio_path, sample_rate, form, **kwargs):
  """
  Wrapper function to load audio in desired form. First raw audio is loaded then if param `form` transformations ops
  are performed (to amplitude,power or mel spectrogram). 
  
  Specific arguments are passed via `kwargs`
  
  :param:
    audio_path (str) : path to audio
    form (str) : representation. Choices = (`ampl`,`power`,`mel`)
  :return:
    audio (np.ndarray) : audio example
    
  """
  
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