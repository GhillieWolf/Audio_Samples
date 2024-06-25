# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 02:35:52 2024

@author: ghill
"""

import os
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import square

# Stwórz katalog, jeśli nie istnieje
przesuniecie = False
katalog = 'audio_data'
if przesuniecie:
    katalog = 'audio_data_przesuniecie'

os.makedirs(katalog, exist_ok=True)

# Funkcja do generowania dźwięków prostokątnych i zapisywania ich jako pliki WAV
def generate_audio_data(filename, frequency, duration, amplitude=0.7, sample_rate=44100, shift_=0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Przesunięcie fazy sygnału
    phase_shift = shift_ / sample_rate
    t_shifted = t + phase_shift
    
    signal = amplitude * square(2 * np.pi * frequency * t_shifted)
    
    write(filename, sample_rate, np.int16(signal * 32767))

frequencies = [
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, # C3 - B3
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, # C4 - B4
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77  # C5 - B5
]

# Generowanie i zapisywanie 6000 plików dźwiękowych
shift = 0
amplitude = 0.7  # Początkowa wartość amplitudy
for i in range(6000):
    label = i % 36  # Etykieta dla danej częstotliwości
    filename = os.path.join(katalog, f'sound_{i}.wav')
    frequency = frequencies[label]
    
    
    
    generate_audio_data(filename, frequency, duration=0.2, amplitude=amplitude, shift_=shift)
    
    if i % 36 == 35:  # Zwiększ przesunięcie fazy co 36 plików
        shift += 1
        amplitude = np.random.uniform(0.1, 0.9) # Co 36 plików losuj nową amplitudę
