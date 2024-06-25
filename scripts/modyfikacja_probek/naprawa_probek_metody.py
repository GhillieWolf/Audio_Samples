# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 02:16:11 2024

@author: ghill
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile

def calculate_lengths(data, threshold=0):
    """Calculate lengths of up and down segments."""
    above = data > threshold
    below = data < threshold

    up_lengths = []
    down_lengths = []

    up_length = 0
    down_length = 0
    for i in range(1, len(data)):
        if above[i] and not above[i-1]:  # start of up
            if down_length > 0:
                down_lengths.append(down_length)
            down_length = 0
        elif below[i] and not below[i-1]:  # start of down
            if up_length > 0:
                up_lengths.append(up_length)
            up_length = 0
        
        if above[i]:
            up_length += 1
        if below[i]:
            down_length += 1
    
    # add the last segment
    if up_length > 0:
        up_lengths.append(up_length)
    if down_length > 0:
        down_lengths.append(down_length)
    
    return up_lengths, down_lengths

def main(directory='pliki', output_directory='pliki2', method='mean2'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    methodes = [
        'mean',
        'mean2',
        'median',
        'median2',
        'max',
        'max2'
    ]

    for methode in methodes:
        method = methode
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                rate, data = wavfile.read(filepath)
                
                if data.ndim > 1:
                    data = data[:, 0]  # use the first channel if stereo
    
                # Calculate lengths of up and down segments
                up_lengths, down_lengths = calculate_lengths(data)
                if filename == "probka_356_F4_1.wav":
                    print("up_lengths")
                    print(up_lengths)
                    print(int(np.mean(up_lengths)))
                    print(int(np.median(up_lengths)))
                    print("down_lengths")
                    print(down_lengths)
                    print(int(np.mean(down_lengths)))
                    print(int(np.median(down_lengths)))
                    
                if method == 'mean':
                    selected_length = int(np.mean(down_lengths))
                elif method == 'median':
                    selected_length = int(np.median(down_lengths))
                elif method == 'max':
                    selected_length_ups = int(np.max(up_lengths))
                    selected_length_downs = int(np.max(down_lengths))
                elif method == 'max2':
                    selected_length = int((np.max(up_lengths) + np.max(down_lengths)) / 2)
                elif method == 'mean2':
                    selected_length = int((np.mean(up_lengths) + np.mean(down_lengths)) / 2)
                elif method == 'median2':
                    selected_length = int((np.median(up_lengths) + np.median(down_lengths)) / 2)
                else:
                    raise ValueError("Method must be 'mean', 'median', 'max', 'max2', 'mean2', or 'median2'")
                
                # Create new waveform
                new_data = np.array([], dtype=data.dtype)
                if method == 'max':
                    toggle = True
                    length = 0
                    while length < len(data):
                        if toggle:
                            new_segment = np.ones(selected_length_ups) * np.max(data)
                            length += selected_length_ups
                        else:
                            new_segment = np.ones(selected_length_downs) * np.min(data)
                            length += selected_length_downs
                        new_data = np.append(new_data, new_segment)
                        toggle = not toggle
                else:
                    for length in range(0, len(data), selected_length):
                        new_segment = np.ones(selected_length) * (np.max(data) if length % (2 * selected_length) < selected_length else np.min(data))
                        new_data = np.append(new_data, new_segment)
                
                # Ensure new data is the same length as original
                new_data = new_data[:len(data)]
                
                # Save the new file
                os.makedirs(output_directory+method, exist_ok=True)
                output_filepath = os.path.join(output_directory+method, filename)
                wavfile.write(output_filepath, rate, new_data.astype(data.dtype))

if __name__ == "__main__":
    main()
