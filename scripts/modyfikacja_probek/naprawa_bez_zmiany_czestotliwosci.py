# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 02:23:01 2024

@author: ghill
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile

def process_wav_file(input_filepath, output_filepath):
    rate, data = wavfile.read(input_filepath)

    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if stereo

    processed_data = np.zeros_like(data, dtype=np.float32)

    for i in range(len(data)):
        if data[i] > 0:
            processed_data[i] = 0.7
        elif data[i] < 0:
            processed_data[i] = -0.7
        elif i > 0 and data[i] == 0:
            if data[i-1] > 0:
                processed_data[i] = -0.7  # falling to zero
            elif data[i-1] < 0:
                processed_data[i] = 0.7   # rising to zero

    # Ensure the processed data is in the same range as the original data
    max_value = np.iinfo(data.dtype).max
    min_value = np.iinfo(data.dtype).min
    processed_data = np.clip(processed_data * max_value, min_value, max_value).astype(data.dtype)
    
    wavfile.write(output_filepath, rate, processed_data)

def main(input_directory='pliki', output_directory='pliki2'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename)
            process_wav_file(input_filepath, output_filepath)

if __name__ == "__main__":
    main()
