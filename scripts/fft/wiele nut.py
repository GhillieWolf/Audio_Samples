# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:36:13 2024

@author: ghill
"""

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read
from scipy.fft import fft
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Definicja nuty
Note = namedtuple('Note', ['name', 'frequency'])

# Lista nut pianina i ich częstotliwości
piano_notes = [
    Note("A0", 27.50), Note("A#0", 29.14), Note("B0", 30.87),
    Note("C1", 32.70), Note("C#1", 34.65), Note("D1", 36.71), Note("D#1", 38.89), Note("E1", 41.20), Note("F1", 43.65), Note("F#1", 46.25), Note("G1", 49.00), Note("G#1", 51.91), Note("A1", 55.00), Note("A#1", 58.27), Note("B1", 61.74),
    Note("C2", 65.41), Note("C#2", 69.30), Note("D2", 73.42), Note("D#2", 77.78), Note("E2", 82.41), Note("F2", 87.31), Note("F#2", 92.50), Note("G2", 98.00), Note("G#2", 103.83), Note("A2", 110.00), Note("A#2", 116.54), Note("B2", 123.47),
    Note("C3", 130.81), Note("C#3", 138.59), Note("D3", 146.83), Note("D#3", 155.56), Note("E3", 164.81), Note("F3", 174.61), Note("F#3", 185.00), Note("G3", 196.00), Note("G#3", 207.65), Note("A3", 220.00), Note("A#3", 233.08), Note("B3", 246.94),
    Note("C4", 261.63), Note("C#4", 277.18), Note("D4", 293.66), Note("D#4", 311.13), Note("E4", 329.63), Note("F4", 349.23), Note("F#4", 369.99), Note("G4", 392.00), Note("G#4", 415.30), Note("A4", 440.00), Note("A#4", 466.16), Note("B4", 493.88),
    Note("C5", 523.25), Note("C#5", 554.37), Note("D5", 587.33), Note("D#5", 622.25), Note("E5", 659.25), Note("F5", 698.46), Note("F#5", 739.99), Note("G5", 783.99), Note("G#5", 830.61), Note("A5", 880.00), Note("A#5", 932.33), Note("B5", 987.77),
    Note("C6", 1046.50), Note("C#6", 1108.73), Note("D6", 1174.66), Note("D#6", 1244.51), Note("E6", 1318.51), Note("F6", 1396.91), Note("F#6", 1479.98), Note("G6", 1567.98), Note("G#6", 1661.22), Note("A6", 1760.00), Note("A#6", 1864.66), Note("B6", 1975.53),
    Note("C7", 2093.00), Note("C#7", 2217.46), Note("D7", 2349.32), Note("D#7", 2489.02), Note("E7", 2637.02), Note("F7", 2793.83), Note("F#7", 2959.96), Note("G7", 3135.96), Note("G#7", 3322.44), Note("A7", 3520.00), Note("A#7", 3729.31), Note("B7", 3951.07),
    Note("C8", 4186.01)
]

# Funkcja do znajdowania najbliższej nuty do danej częstotliwości
def find_closest_note(frequency):
    return min(piano_notes, key=lambda note: abs(note.frequency - frequency))

# Wczytaj próbkę dźwiękową
sample_rate, samples = read('piano.wav')

# Sprawdź, czy próbka jest mono czy stereo
if len(samples.shape) == 2:
    samples = samples.mean(axis=1)  # Konwersja do mono przez uśrednienie kanałów

# Ustawienia okna czasowego
window_duration = 0.2  # w sekundach
window_size = int(window_duration * sample_rate)

# Przesunięcie początkowe okna (0 - początek pliku, 1 - miejsce gdzie zaczyna się normalnie drugie okno)
initial_window_offset = 1.5  # Przesunięcie jako część okna 0 albo 1.5 działa

# Próg amplitudy
amplitude_threshold = 0.9999

# Parametr długości pliku na wykresie
plot_duration = 5  # Czas w sekundach

# Przetwarzanie segmentów
total_segments = int(plot_duration / window_duration)

# Ustal liczbę segmentów do pominięcia
ignore_last_n_windows = int(initial_window_offset // 1)

# Ustal ostateczną liczbę segmentów do analizy
num_segments = total_segments - ignore_last_n_windows

# Ustal początkowy indeks okna
initial_start_index = int((initial_window_offset % 1) * window_size)

print("Istotne częstotliwości i nuty w kolejności występowania:")

# Listy do przechowywania wyników do wykresu
times = []
frequencies = []
window_times = []
notes = []

for i in range(num_segments):
    start_index = initial_start_index + i * window_size
    end_index = start_index + window_size
    segment = samples[start_index:end_index]
    
    # Przeprowadź transformatę Fouriera
    fft_result = fft(segment)
    
    # Oblicz amplitudę dla każdej częstotliwości
    amplitude = np.abs(fft_result)
    
    # Ustaw próg amplitudy
    max_amplitude = np.max(amplitude)
    significant_indices = np.where(amplitude > max_amplitude * amplitude_threshold)[0]
    
    # Oblicz częstotliwości dla tych indeksów
    significant_frequencies = sample_rate * significant_indices / len(segment)
    
    # Filtruj częstotliwości i amplitudy powyżej progu
    significant_amplitudes = amplitude[significant_indices]
    significant_frequencies = significant_frequencies[significant_amplitudes.argsort()[::-1]]  # Sortowanie według amplitudy malejąco
    
    # Wyświetl istotne częstotliwości i odpowiadające im nuty
    for freq in significant_frequencies:
        closest_note = find_closest_note(freq)

        if int(closest_note.frequency) > 4100:
            print(f"Częstotliwość pomijam: {freq:.2f} Hz - Nuta: {closest_note.name} ({closest_note.frequency:.2f} Hz)")
            continue
        
        print(f"Częstotliwość: {freq:.2f} Hz - Nuta: {closest_note.name} ({closest_note.frequency:.2f} Hz)")
        
        # Dodaj dane do wykresu
        times.append(start_index / sample_rate)
        frequencies.append(freq)
        notes.append(closest_note.name)

        # Odtwarzaj znalezione nuty w głośniku
        duration = 0.2  # Czas trwania nuty w sekundach (tutaj 0.2 sekundy)
        #sd.play(np.sin(2 * np.pi * closest_note.frequency * np.linspace(0, duration, int(duration * sample_rate))), samplerate=sample_rate)
        #sd.wait()
    
    # Dodaj czas okna do listy
    window_times.append((start_index / sample_rate, end_index / sample_rate))

# Rysuj wykres
fig, ax1 = plt.subplots(figsize=(12, 6))

# Wykres waveform na osi pierwszej (lewej)
time_axis = np.linspace(0, plot_duration, num=len(samples[:plot_duration * sample_rate]))
ax1.plot(time_axis, samples[:plot_duration * sample_rate], color='gray', alpha=0.5)
ax1.set_xlabel('Czas (s)')
ax1.set_ylabel('Amplituda', color='gray')
ax1.tick_params(axis='y', labelcolor='gray')

# Dodaj prostokąty oznaczające okna czasowe
for (start_time, end_time) in window_times:
    rect = patches.Rectangle((start_time, ax1.get_ylim()[0]), end_time - start_time, ax1.get_ylim()[1] - ax1.get_ylim()[0], linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)

# Druga oś y dla częstotliwości
ax2 = ax1.twinx()
ax2.plot(times, frequencies, 'bo')
ax2.set_ylabel('Częstotliwość (Hz)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Dodaj etykiety do częstotliwości
for t, f, n in zip(times, frequencies, notes):
    ax2.text(t, f, n, fontsize=9, color='blue', ha='right')

plt.title('Częstotliwości wykryte w czasie trwania melodi przesunięcie okna o 0.5')
plt.show()
