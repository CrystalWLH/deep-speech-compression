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


def normalize(audio):
  """
  Normalize audio signal. Mean 0 std 1
  
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

def load_raw_audio(audio_path , sample_rate):
  """
  Load single audio file from path.
  
  :param:
    audio_path (str) : path to audio
    sample_rate (int) : rate at which audio is sampled
  :return:
    raw_audio (np.ndarray) : audio examples (wave)
  """
  try:
    raw_audio = librosa.load(path = audio_path, sr = sample_rate)[0]
    return normalize(raw_audio[np.newaxis,:])
  except IndexError:
    logger.warning("No audio loaded for this file : `{}`".format(audio_path))
    pass
  
  
def wave2mfccs(audio_path, sample_rate = 16000, n_mfcc=13, n_fft=512, hop_length=160):
  """
  Calculate mfcc coefficients from the given raw audio data
  Args:
    audio_data: numpyarray of raw audio wave
    samplerate: the sample rate of the `audio_data`
    n_mfcc: the number of coefficients to generate
    n_fft: the window size of the fft
    hop_length: the hop length for the window
  Returns: the mfcc coefficients in the form [time, coefficients]
  """
  
  audio = load_raw_audio(audio_path = audio_path, sample_rate = sample_rate)
  
  mfcc = librosa.feature.mfcc(y = audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

  # add derivatives and normalize
  mfcc_delta = librosa.feature.delta(mfcc)
  mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
  mfcc = np.concatenate((normalize(mfcc),
                         normalize(mfcc_delta),
                         normalize(mfcc_delta2)), axis=0)

  return mfcc
  

def wave2power_spectrogram(audio_path, sample_rate, n_fft=512, hop_length=160):
  """
  Compute power spectrogram from wave audio.
  
  :param:
    audio (np.ndarray) : audio examples (wave)
    fft_window (int) : window for FFT
    hop_length (int) : number audio of frames between FFT columns
  :return:
    spectrogram (np.ndarray) : Power spectrogram
    
  """
  
  audio = load_raw_audio(audio_path = audio_path, sample_rate = sample_rate)
  
  spectogram = librosa.stft(y=audio, n_fft= n_fft, hop_length= hop_length)
  
  return normalize(np.abs(spectogram) ** 2)


def get_audio(audio_path, sample_rate = 16000, form = 'mfccs', n_fft = 400, hop_length = 160, n_mfcc = 40):
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
  
  if form == 'raw':
    
    audio = load_raw_audio(audio_path = audio_path, sample_rate = sample_rate)
  
  elif form == 'power':
  
   audio = wave2power_spectrogram(audio_path = audio_path,sample_rate = sample_rate,n_fft = n_fft, hop_length = hop_length)
    
  elif form == 'mfccs':
    
    audio = wave2mfccs(audio_path = audio_path,sample_rate = sample_rate,n_fft = n_fft, hop_length = hop_length, n_mfcc = n_mfcc)
                                 
  return audio


if __name__ == "__main__":
  
  test_raw = ('LibriSpeech/dev-other/6455/66379/6455-66379-0011.flac', 16000, 'raw', 512, 160, 13)
  
  test_spec = ('LibriSpeech/dev-other/6455/66379/6455-66379-0011.flac', 16000, 'power', 512, 160, 13)
  
  test_mfccs = ('LibriSpeech/dev-other/6455/66379/6455-66379-0011.flac', 16000, 'mfccs', 512, 160, 13)
  
  raw = get_audio(*test_raw)
  
  print(raw.shape)
  
  spec = get_audio(*test_spec)
  
  print(spec.shape)
  
  mfccs = get_audio(*test_mfccs)
  
  print(mfccs.shape)