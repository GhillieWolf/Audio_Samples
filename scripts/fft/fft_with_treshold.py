# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:10:20 2024

@author: ghill
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# Funkcja do znalezienia częstotliwości powyżej zadanego progu
def find_frequencies_above_threshold(filename, threshold_percentage, max_frequencies=10):
    # Wczytywanie pliku WAV
    sampling_rate, data = wavfile.read(filename)

    # Normalizacja danych audio (jeśli dane są typu int)
    if data.dtype == np.int16:
        data = data / 32768.0

    # Sprawdzenie, czy dane są wielokanałowe (stereo)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Zamiana na mono poprzez uśrednienie kanałów

    # Wykonywanie FFT
    N = len(data)
    data_fft = fft(data)
    
    # Obliczanie amplitudy sygnału i normalizacja
    amplitudes = np.abs(data_fft) / N
    
    # Obliczanie częstotliwości
    freqs = fftfreq(N, 1/sampling_rate)
    
    # Ustalanie progu amplitudy
    threshold = threshold_percentage * np.max(amplitudes)
    
    # Filtracja częstotliwości powyżej progu i usunięcie ujemnych częstotliwości
    positive_freqs = freqs[:N//2]
    positive_amplitudes = amplitudes[:N//2]
    filtered_freqs = positive_freqs[positive_amplitudes > threshold]
    filtered_amplitudes = positive_amplitudes[positive_amplitudes > threshold]
    
    # Sortowanie częstotliwości i amplitud malejąco po amplitudzie
    sorted_indices = np.argsort(filtered_amplitudes)[::-1]
    filtered_freqs = filtered_freqs[sorted_indices][:max_frequencies]
    filtered_amplitudes = filtered_amplitudes[sorted_indices][:max_frequencies]
    
    return positive_freqs, positive_amplitudes, threshold, filtered_freqs, filtered_amplitudes

# Parametry
filename = 'sin_440hz.wav'  # Zamień na nazwę swojego pliku WAV
filename= "probka_313_D#4_2.wav"
filename= "probka_830_G#5_1.wav"
threshold_percentage = 0.10  # Ustawienie progu na 10%

# Znalezienie częstotliwości powyżej progu
frequencies, amplitudes, threshold, filtered_freqs, filtered_amplitudes = find_frequencies_above_threshold(filename, threshold_percentage)

# Tworzenie wykresu
fig, ax = plt.subplots(figsize=(10, 6))

# Wykres amplitudy
ax.plot(frequencies, amplitudes, label='Amplituda')
ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentage*100}%)')
ax.set_title('Częstotliwości powyżej progu amplitudy')
ax.set_xlabel('Częstotliwość [Hz]')
ax.set_ylabel('Amplituda')
ax.set_xscale('log')  # Ustawienie osi częstotliwości logarytmicznie
ax.legend()
ax.grid()

# Dodanie tekstu pod wykresem (w górnym lewym rogu)
textstr = '\n'.join([f"Częstotliwość: {freq:.2f} Hz, Amplituda: {amp:.4f}" for freq, amp in zip(filtered_freqs, filtered_amplitudes)])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()