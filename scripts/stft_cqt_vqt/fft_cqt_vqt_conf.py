import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from tabulate import tabulate




# Example usage
#file_dir_path = 'probki_multiplayerpiano/'  # Replace with your audio file path
file_dir_path = 'probki_roland/'  # Replace with your audio file path

#file_dir_path = 'probki_synt'
#file_dir_path = 'probki_multiplayerpiano_przyciete'
#file_dir_path = 'probki_roland_przyciete_prostakatne'
#file_dir_path = 'probki_multiplayerpiano_przyciete_prostokatne'
#file_dir_path = 'probki_roland_przyciete'


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def perform_fft(y, sr):
    D = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    
    freqs = librosa.fft_frequencies(sr=sr)
    midi_vals = librosa.hz_to_midi(freqs)
    note_names = []
    for midi in midi_vals:
        try:
            note_names.append(librosa.midi_to_note(int(np.round(midi))))
        except OverflowError:
            note_names.append('NaN')

    max_amplitude_frame = np.argmax(np.sum(D, axis=0))
    note_idx = np.argmax(D[:, max_amplitude_frame])
    detected_note = note_names[note_idx]
    max_amplitude_time = librosa.frames_to_time([max_amplitude_frame], sr=sr)[0]

    # Sprawdzenie, czy nuta jest poza zakresem
    if freqs[note_idx] <= librosa.note_to_hz('B2'):
        detected_note = '<C3'
    elif freqs[note_idx] >= librosa.note_to_hz('C6'):
        detected_note = '>B5'

    return S_db, max_amplitude_time, detected_note, freqs

def perform_cqt(y, sr):
    C = np.abs(librosa.cqt(y, sr=sr))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    
    cqt_freqs = librosa.cqt_frequencies(C_db.shape[0], fmin=librosa.note_to_hz('C1'))
    cqt_note_names = []
    for freq in cqt_freqs:
        try:
            cqt_note_names.append(librosa.hz_to_note(freq))
        except ValueError:
            cqt_note_names.append('NaN')

    max_amplitude_frame = np.argmax(np.sum(C, axis=0))
    note_idx = np.argmax(C[:, max_amplitude_frame])
    detected_note = cqt_note_names[note_idx]
    max_amplitude_time = librosa.frames_to_time([max_amplitude_frame], sr=sr)[0]

    # Sprawdzenie, czy nuta jest poza zakresem
    if cqt_freqs[note_idx] <= librosa.note_to_hz('B2'):
        detected_note = '<C3'
    elif cqt_freqs[note_idx] >= librosa.note_to_hz('C6'):
        detected_note = '>B5'

    return C_db, max_amplitude_time, detected_note

def perform_vqt(y, sr):
    gamma = 10
    C_vqt = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=12, n_bins=84, tuning=None, filter_scale=gamma))
    C_db_vqt = librosa.amplitude_to_db(C_vqt, ref=np.max)
    
    vqt_freqs = librosa.cqt_frequencies(C_db_vqt.shape[0], fmin=librosa.note_to_hz('C1'))
    vqt_note_names = []
    for freq in vqt_freqs:
        try:
            vqt_note_names.append(librosa.hz_to_note(freq))
        except ValueError:
            vqt_note_names.append('NaN')

    max_amplitude_frame = np.argmax(np.sum(C_vqt, axis=0))
    note_idx = np.argmax(C_vqt[:, max_amplitude_frame])
    detected_note = vqt_note_names[note_idx]
    max_amplitude_time = librosa.frames_to_time([max_amplitude_frame], sr=sr)[0]

    # Sprawdzenie, czy nuta jest poza zakresem
    if vqt_freqs[note_idx] <= librosa.note_to_hz('B2'):
        print(f'{vqt_freqs[note_idx]} na <C3')
        detected_note = '<C3'
    elif vqt_freqs[note_idx] >= librosa.note_to_hz('C6'):
        print(f'{vqt_freqs[note_idx]} na >B5')
        detected_note = '>B5'

    return C_db_vqt, max_amplitude_time, detected_note

def main(file_dir_path):
    detected_notes_fft_all = {}
    detected_notes_cqt_all = {}
    detected_notes_vqt_all = {}

    for filename in os.listdir(file_dir_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(file_dir_path, filename)
            print(f"Processing {filename}...")
            
            y, sr = load_audio(file_path)
            
            S_db_fft, onset_time_fft, detected_note_fft, freqs = perform_fft(y, sr)
            detected_notes_fft_all[filename] = (onset_time_fft, detected_note_fft)
            
            C_db_cqt, onset_time_cqt, detected_note_cqt = perform_cqt(y, sr)
            detected_notes_cqt_all[filename] = (onset_time_cqt, detected_note_cqt)
            
            C_db_vqt, onset_time_vqt, detected_note_vqt = perform_vqt(y, sr)
            detected_notes_vqt_all[filename] = (onset_time_vqt, detected_note_vqt)

    return detected_notes_fft_all, detected_notes_cqt_all, detected_notes_vqt_all

import seaborn as sns
import re
"""
def confusion_matrix2(true_freqs, predicted_freqs, labels=None):
    if labels is None:
        labels = sorted(set(true_freqs) | set(predicted_freqs))

    label_to_index = {label: index for index, label in enumerate(labels)}

    # Inicjalizujemy macierz pomyłek zerami
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Zliczamy wystąpienia par (rzeczywista, przewidziana)
    for true, pred in zip(true_freqs, predicted_freqs):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index][pred_index] += 1

    return matrix
"""
def create_confusion_matrix(true_freqs, predicted_freqs):
    labels = np.unique(true_freqs)
    sorted_keys = sort_by_piano_order_with_octaves(labels)
    
    print(f'labels {len(sorted_keys)}')
    print(sorted_keys)
    cm = confusion_matrix(true_freqs, predicted_freqs, labels=sorted_keys)
    suma = np.sum(cm, axis=None)
    print(f'cm {suma}')
    print(predicted_freqs)
    print(true_freqs)
    print(f'true_freqs {len(true_freqs)}')
    print(f'predicted_freqs {len(predicted_freqs)}')
    polaczone = np.unique(np.concatenate((true_freqs, predicted_freqs)))
    print(f'predicted_freqs i true_freqs {len(polaczone)}')
    print(polaczone)
    return cm

# Funkcja do rysowania macierzy pomyłek jako mapy cieplnej
def plot_confusion_matrix(conf_matrix, labels, text):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Obliczenie liczby poprawnie odgadniętych przypadków
    correct = np.trace(conf_matrix)
    # Obliczenie liczby źle odgadniętych przypadków
    total = np.sum(conf_matrix)
    incorrect = total - correct
    print("Poprawnie odgadnięte przypadki:", correct)
    print("Źle odgadnięte przypadki:", incorrect)
    if file_dir_path == 'probki_radde/':
        plt.title(f'Dane Roland {text}, wszystkie: {total}, niepoprawne: {incorrect}, poprawne:{correct}' ) #nagrane mikrofonem z midi keyboard Roland
    if file_dir_path == 'probki_moje/':
        plt.title(f'Dane MultiplayerPiano {text}, wszystkie: {total}, niepoprawne: {incorrect}, poprawne:{correct}' )
    if file_dir_path == 'probki_synt/':
        plt.title(f'Dane syntetyczne {text}, wszystkie: {total}, niepoprawne: {incorrect}, poprawne:{correct}' )
    plt.show()
       
def sort_by_piano_order_with_octaves(notes):
    piano_order = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]
    
    # Stwórzmy mapę z indeksem każdego dźwięku dla szybkiego dostępu
    piano_index = {note: index for index, note in enumerate(piano_order)}
    
    def note_key(note):
        # Wyodrębnij dźwięk i numer oktawy
        if note == '<C3':
            return (-1, -1)
        elif note == '>B5':
            return (10, 10)
        match = re.match(r"([A-G]#?)(\d+)", note)
        if not match:
            raise ValueError(f"Invalid note format: {note}")
        note_part, octave_part = match.groups()
        octave = int(octave_part)
        return (octave, piano_index[note_part])
    
    # Użyjmy klucza sortującego, który zamienia dźwięk i oktawę na krotkę (octave, index)
    sorted_notes = sorted(notes, key=note_key)
    
    # Dodanie na początku
    sorted_notes = np.insert(sorted_notes, 0, '<C3')
    # Dodanie na końcu
    sorted_notes = np.append(sorted_notes, '>B5')
    
    return sorted_notes

def display_results(detected_notes_fft_all, detected_notes_cqt_all, detected_notes_vqt_all):
    true_notes = []
    fft_notes = []
    cqt_notes = []
    vqt_notes = []

    for filename, (onset_time_fft, detected_note_fft) in detected_notes_fft_all.items():
        onset_time_cqt, detected_note_cqt = detected_notes_cqt_all[filename]
        onset_time_vqt, detected_note_vqt = detected_notes_vqt_all[filename]

        true_note = filename.split('_')[2]
        true_notes.append(true_note)
        fft_notes.append(detected_note_fft)
        cqt_notes.append(detected_note_cqt)
        vqt_notes.append(detected_note_vqt)
        
    fft_notes = np.char.replace(fft_notes, '♯', '#')
    cqt_notes = np.char.replace(cqt_notes, '♯', '#')
    vqt_notes = np.char.replace(vqt_notes, '♯', '#')
    
    confusion_matrix_fft = create_confusion_matrix(true_notes, fft_notes)
    confusion_matrix_cqt = create_confusion_matrix(true_notes, cqt_notes)
    confusion_matrix_vqt = create_confusion_matrix(true_notes, vqt_notes)
    
     
    sorted_keys = sort_by_piano_order_with_octaves(np.unique(true_notes))
    
    
    plot_confusion_matrix(confusion_matrix_fft, labels=sorted_keys, text="STFT")
    plot_confusion_matrix(confusion_matrix_cqt, labels=sorted_keys, text="CQT")
    plot_confusion_matrix(confusion_matrix_vqt, labels=sorted_keys, text="VQT")

    print("STFT Confusion Matrix:")
    print(confusion_matrix_fft)
    print("\nCQT Confusion Matrix:")
    print(confusion_matrix_cqt)
    print("\nVQT Confusion Matrix:")
    print(confusion_matrix_vqt)

    headers = ['Filename', 'FFT Time', 'FFT Note', 'CQT Time', 'CQT Note', 'VQT Time', 'VQT Note']
    table_data = []
    
    for filename, (onset_time_fft, detected_note_fft) in detected_notes_fft_all.items():
        onset_time_cqt, detected_note_cqt = detected_notes_cqt_all[filename]
        onset_time_vqt, detected_note_vqt = detected_notes_vqt_all[filename]

        table_data.append([filename, onset_time_fft, detected_note_fft, onset_time_cqt, detected_note_cqt, onset_time_vqt, detected_note_vqt])

    df = pd.DataFrame(table_data, columns=headers)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    

detected_notes_fft_all, detected_notes_cqt_all, detected_notes_vqt_all = main(file_dir_path)
display_results(detected_notes_fft_all, detected_notes_cqt_all, detected_notes_vqt_all)
