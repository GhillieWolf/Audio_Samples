# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 01:18:21 2024

@author: ghill
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from scipy.signal.windows import gaussian
import cv2

# Ustawienia
katalog = 'audio_data'
przesuniecie = True
wygladzenieLevel = 1.5
skaluj = True
specSize = 28
if przesuniecie:
    katalog = 'audio_data_przesuniecie'

frequencies = [
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, # C3 - B3
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, # C4 - B4
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77  # C5 - B5
]

# Wczytanie dźwięków i utworzenie tablicy NumPy dla danych obrazowych oraz etykiet
image_data = []
labels = []

def stretch_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    stretched_arr = (arr - min_val) * (255 / (max_val - min_val))
    return stretched_arr

print("Generowanie bez wygładzania:")
for i in range(6000):
    label = i % 36
    filename = os.path.join(katalog, f'sound_{i}.wav')
    sample_rate, samples = read(filename)
    _, _, Pxx = spectrogram(samples, fs=sample_rate)
    normalized_spectrogram = np.log1p(np.abs(Pxx))
    if skaluj:
        resized_spectrogram = cv2.resize(normalized_spectrogram, (specSize, specSize), interpolation=cv2.INTER_LANCZOS4)
    else:
        resized_spectrogram = normalized_spectrogram
    rozmiarx = resized_spectrogram.shape[0]
    rozmiary = resized_spectrogram.shape[1]
    image_data.append(resized_spectrogram)
    labels.append(label)
    if i == 0:
        plt.imshow(resized_spectrogram, cmap='gray')
        plt.title(f'{i} freq: {label}: {frequencies[label]}Hz')
        plt.axis('off')
        plt.show()
        
image_data = np.array(image_data)
labels = np.array(labels)

# Konwertuj image_data do formatu float32 i znormalizuj do zakresu [0, 1]
image_data = image_data.astype('float32') / 255

# Podziel dane na zestaw treningowy i testowy
split = int(0.8 * len(image_data))
x_train, y_train = image_data[:split], labels[:split]
x_test, y_test = image_data[split:], labels[split:]

# Zmodyfikuj kształt danych dla zgodności z modelem CNN
x_train = x_train.reshape((-1, rozmiarx, rozmiary, 1))
x_test = x_test.reshape((-1, rozmiarx, rozmiary, 1))

# Zmodyfikuj etykiety na kategorialne
y_train = to_categorical(y_train, num_classes=36)
y_test = to_categorical(y_test, num_classes=36)

# Zbuduj model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(rozmiarx, rozmiary, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(36, activation='softmax')
])

# Skompiluj model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenuj model
hist = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Ocena modelu
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Zapisz model w formacie natywnym Keras
model_filename = 'audio_model_rozmiar{}.keras'.format(specSize)
if przesuniecie:
    model_filename = f'audio_model_przesuniecie_rozmiar{specSize}.keras'
print(f'{model_filename} Test accuracy:', test_acc)
model.save(model_filename)

# Lista nut i częstotliwości
piano_keys = [
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
]

# Generowanie wykresów dla losowych próbek testowych
num_samples = 5
random_indices = np.random.choice(len(x_test), num_samples, replace=False)

for idx in random_indices:
    spectrogram_image = x_test[idx].reshape(x_train.shape[1], x_train.shape[2])
    predicted_label = np.argmax(model.predict(x_test[idx].reshape(1, x_train.shape[1], x_train.shape[2], 1)))
    true_label = np.argmax(y_test[idx])
    
    plt.imshow(spectrogram_image, cmap='gray')
    plt.title(f'idx: {idx} True Label: {piano_keys[true_label]}, Predicted: {piano_keys[predicted_label]}')
    plt.axis('off')
    plt.show()
    #print(x_test[idx].reshape(1, x_train.shape[1], x_train.shape[2], 1).shape)
    #print(x_test[idx].reshape(1, x_train.shape[1], x_train.shape[2], 1))
    
    

# Wizualizacja wyników
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Ewaluacja za pomocą precyzji, recall i accuracy
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

yhat = model.predict(x_test)
pre.update_state(y_test, yhat)
re.update_state(y_test, yhat)
acc.update_state(y_test, yhat)

print('Precision:', pre.result().numpy())
print('Recall:', re.result().numpy())
print('Binary Accuracy:', acc.result().numpy())

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# Ewaluacja za pomocą precyzji, recall i accuracy ze scikit-learn
yhat = model.predict(x_test)
yhat_classes = np.argmax(yhat, axis=1)
y_true = np.argmax(y_test, axis=1)

precision = precision_score(y_true, yhat_classes, average='macro')
recall = recall_score(y_true, yhat_classes, average='macro')
f1 = f1_score(y_true, yhat_classes, average='macro')
conf_matrix = confusion_matrix(y_true, yhat_classes)

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_matrix)