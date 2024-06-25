# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:50:12 2024

@author: ghill
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re
import cv2

# Wczytaj model
specSize = 28
przesuniecie = True
wygladzenie = False

folders = [
    #'pliki2max2',
    #'pliki2max',
    #'pliki2mean',
    #'pliki2mean2',
    #'pliki2median',
    #'pliki2median2',
    'probki_multiplayerpiano',
    'probki_synt',
    'probki_multiplayerpiano_przyciete',
    'probki_roland_przyciete',
    'probki_roland_przyciete_prostakatne',
    'probki_multiplayerpiano_przyciete_prostokatne'
]

for folder in folders:
    folder_path = folder
    print(f'Przetwarzanie folderu: {folder}')
    if wygladzenie and przesuniecie:
        print(f'audio_model_wygladzenie_przesuniecie_rozmiar{specSize} load')
        model = load_model(f'audio_model_wygladzenie_przesuniecie_rozmiar{specSize}.keras')
    elif wygladzenie:
        print(f'audio_model_wygladzenie_rozmiar{specSize} load')
        model = load_model(f'audio_model_wygladzenie_rozmiar{specSize}.keras')
    elif przesuniecie:
        print(f'audio_model_przesuniecie_rozmiar{specSize} load')
        model = load_model(f'audio_model_przesuniecie_rozmiar{specSize}.keras')
    else:
        print(f'audio_model_rozmiar{specSize} load')
        model = load_model(f'audio_model_rozmiar{specSize}.keras')
    
    
    # Definiowanie listy częstotliwości
    frequencies = [
        130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, # C3 - B3
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, # C4 - B4
        523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77  # C5 - B5
    ]
    # Tablica nazw klawiszy
    keys_ = [
        "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
        "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
        "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
    ]
    
    def get_key_name(frequency):
        frequencies_ = frequencies
        if frequency in frequencies_:
            index = frequencies_.index(frequency)
            return keys_[index]
        else:
            return "Frequency not found"
        
        
    # Funkcja do znajdowania najbliższej częstotliwości
    def find_closest_frequency(target_freq, frequencies):
        closest_freq = min(frequencies, key=lambda x: abs(x - target_freq))
        return closest_freq
    
    # Funkcja do ekstrakcji częstotliwości z nazwy pliku
    def extract_frequency(filename):
        pattern = r'probka_(\d+)_([A-Za-z#]+\d+)_(\d+)\.wav'
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f'Nazwa pliku {filename} nie pasuje do oczekiwanego wzorca.')
    
    # Funkcja do ekstrakcji nazwy klawisza z nazwy pliku
    def extract_keyname(filename):
        pattern = r'probka_(\d+)_([A-Za-z#]+\d+)_(\d+)\.wav'
        match = re.search(pattern, filename)
        if match:
            return match.group(2)
        else:
            raise ValueError(f'Nazwa pliku {filename} nie pasuje do oczekiwanego wzorca.')
    
    # Funkcja do ładowania plików z folderu
    def load_files_from_folder(folder):
        files = os.listdir(folder)
        audio_files = [f for f in files if f.endswith('.wav') and os.path.isfile(os.path.join(folder, f))]
        return audio_files
    
    # Funkcja do przewidywania częstotliwości
    def predict_and_show(test_image):
        #smoothed_image = gaussian_filter(test_image, sigma=0.5)
        #test_image_processed = smoothed_image.reshape((1, specSize, specSize, 1))
        test_image_processed = test_image.reshape((1, specSize, specSize, 1))
        predicted_label = np.argmax(model.predict(test_image_processed)[0])
        return predicted_label
    
    # Funkcja do tworzenia macierzy pomyłek
    def create_confusion_matrix(true_freqs, predicted_freqs):
        labels = np.unique(true_freqs)
        cm = confusion_matrix(true_freqs, predicted_freqs, labels=labels)
        return cm
    
    def classify_elements(conf_matrix):
        n = conf_matrix.shape[0]
        correctly_classified = 0
        incorrectly_classified = 0
        close_classified = 0
    
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Correctly classified (diagonal)
                    correctly_classified += conf_matrix[i, j]
                else:
                    # Incorrectly classified (off-diagonal)
                    incorrectly_classified += conf_matrix[i, j]
                    # Close classified (within 2 positions from diagonal)
                    if abs(i - j) <= 2:
                        close_classified += conf_matrix[i, j]
    
        return correctly_classified, incorrectly_classified, close_classified

    # Funkcja do rysowania macierzy pomyłek jako mapy cieplnej
    def plot_confusion_matrix(conf_matrix, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        correct, incorrect, close = classify_elements(conf_matrix)
        
        if folder_path == 'moje_probki': 
            plt.title(f'Confusion Matrix MultiplayerPiano Nieprzycięte, poprawne:{correct}, niepoprawne:{incorrect} bliskie: {close}')
        elif folder_path == 'radde_probki':
            plt.title(f'Confusion Matrix Roland Nieprzycięte, poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        elif folder_path == 'przyciete/synt': 
            plt.title(f'Confusion Matrix Syntetyczne, poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        elif folder_path == 'przyciete/moje':
            plt.title(f'Confusion Matrix MultiplayerPiano, poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        elif folder_path == 'przyciete/radde':
            plt.title(f'Confusion Matrix Roland, poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        elif folder_path == 'przyciete/mojen':
            plt.title(f'Confusion Matrix MultiplayerPiano Poprawione, poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        elif folder_path == 'przyciete/radden':
            plt.title(f'Confusion Matrix Roland Poprawione, poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        else:
            plt.title(f'Confusion Matrix poprawne:{correct}, niepoprawne:{incorrect} blisko: {close}')
        plt.show()
    
    # Wczytaj pliki audio
    audio_files = load_files_from_folder(folder_path)
    true_frequencies = []
    predicted_frequencies = []
    true_key_names = []
    predicted_key_names = []
    
    for file in audio_files:
        try:
            true_frequency = extract_frequency(file)
            true_keyname = extract_keyname(file)
            true_frequencies.append(int(find_closest_frequency(true_frequency, frequencies)))
            kname = get_key_name(find_closest_frequency(true_frequency, frequencies))
            true_key_names.append(kname)
            
            # Przetwarzanie pliku audio
            filename = os.path.join(folder_path, file)
            sample_rate, samples = read(filename)
            
            _, _, Pxx = spectrogram(samples, fs=sample_rate)
            normalized_spectrogram = np.log1p(np.abs(Pxx))
            
            
            
            resized_spectrogram = cv2.resize(normalized_spectrogram, (specSize, specSize), interpolation=cv2.INTER_LANCZOS4)
            spectrogram_image = resized_spectrogram.astype('float32') / 255
            spectrogram_image = spectrogram_image.reshape(1, 28, 28, 1)
            predicted_frequency_index = model.predict(spectrogram_image)
            predicted_frequency_index = np.argmax(predicted_frequency_index)
            
            predicted_frequencies.append(int(frequencies[predicted_frequency_index]))
            predicted_key_names.append(get_key_name(frequencies[predicted_frequency_index]))
            
            """
            plt.imshow(resized_spectrogram, cmap='gray')
            plt.title(f'Plik: {file}, Przewidziana częstotliwość: {frequencies[predicted_frequency_index]} Hz')
            plt.axis('off')
            plt.show()
            print(f'Plik: {file}, Przewidziana częstotliwość: {frequencies[predicted_frequency_index]} Hz')
            """
            
            
            """
            ###
            #file = f"sound_{xx}.wav"
            #filename = os.path.join(folder_path, file)
            sample_rate, samples = read(filename)
            _, _, Pxx = spectrogram(samples, fs=sample_rate)
            normalized_spectrogram = np.log1p(np.abs(Pxx))
            skaluj = True
            if skaluj:
                resized_spectrogram = cv2.resize(normalized_spectrogram, (specSize, specSize), interpolation=cv2.INTER_LANCZOS4)
            else:
                resized_spectrogram = normalized_spectrogram
            resized_spectrogram = resized_spectrogram.astype('float32') / 255
            
            img=resized_spectrogram
            
            spectrogram_image = img.reshape(28, 28)
            predicted_label = np.argmax(model.predict(img.reshape(1, 28, 28, 1)))
            true_label = np.argmax(img)
            
            plt.imshow(spectrogram_image, cmap='gray')
            plt.title(f'moje2 Plik: {file}, {keys_[true_label]} Predicted: {frequencies[predicted_label]}')
            plt.axis('off')
            plt.show()
            
            """
            
        except ValueError as e:
            print(e)
            continue
    
    print("Prawdziwe częstotliwości:", true_frequencies)
    print("Przewidziane częstotliwości:", predicted_frequencies)
    
    # Tworzenie macierzy pomyłek
    conf_matrix = create_confusion_matrix(true_frequencies, predicted_frequencies)
    print("Macierz pomyłek:")
    print(conf_matrix)
    
    # Sortowanie dla macierzy pomyłek
    def sort_by_piano_order_with_octaves(notes):
        piano_order = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
        ]
        
        piano_index = {note: index for index, note in enumerate(piano_order)}
        
        def note_key(note):
            match = re.match(r"([A-G]#?)(\d+)", note)
            if not match:
                raise ValueError(f"Invalid note format: {note}")
            note_part, octave_part = match.groups()
            octave = int(octave_part)
            return (octave, piano_index[note_part])
        
        sorted_notes = sorted(notes, key=note_key)
        
        return sorted_notes
    
    sorted_keys = sort_by_piano_order_with_octaves(np.unique(true_key_names))
    plot_confusion_matrix(conf_matrix, labels=sorted_keys)
