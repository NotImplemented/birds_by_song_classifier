import os
import numpy
import numpy.fft
import struct
import matplotlib.image
import matplotlib.pyplot
import tensorflow
import wave
import fourier
from time import ctime

import time as time_module


output_classes = 90

sample_size = 512
time_shift = int(sample_size * 0.75)
max_song_length = 6.5
max_frame_rate = 44100
max_spectrogram_length = int((max_frame_rate * max_song_length - sample_size) / time_shift + 1)
rows = int(sample_size / 2 - 16)

image_rows = rows
image_columns = max_spectrogram_length
dataset_size = 4096

def prepare_dataset():

    print('[' + ctime() + ']: Data preparation has started.')
    start_time = time_module.time()

    class_offset = 5

    wav_path = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', 'test')
    labels_path = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS', 'nips4b_birdchallenge_train_labels.csv')

    pixel_type = numpy.dtype(numpy.uint16)
    factor = 1.0 / (numpy.iinfo(pixel_type).max - numpy.iinfo(pixel_type).min)

    labels = []
    spectrograms = []

    file_labels = open(labels_path)
    file_labels.readline()
    species = file_labels.readline()
    auxiliary = file_labels.readline()

    print(species)
    print(auxiliary)
    class_names = species.split(',')[class_offset:]
    print(class_names)

    file_index = 0
    display_count = 8
    for file in os.listdir(wav_path):

        file_path = os.path.abspath(os.path.join(wav_path, file))
        _, extension = os.path.splitext(file_path)

        if extension != '.wav':
            continue

        # read labels and set up classes
        line = file_labels.readline()
        classes = line.split(',')
        label = numpy.zeros((1, output_classes))
        description = ''

        for i in range(1, len(classes)):
            if classes[i]:
                label[(0, i - class_offset)] = 1.0
                if description:
                    description += ', '
                description += class_names[i-class_offset]

        if description is None:
            continue

        # read raw sound and build spectrogram
        sound = wave.open(file_path, 'r')
        frames_count = sound.getnframes()
        frame_rate = sound.getframerate()
        sample_width = sound.getsampwidth()
        sound_channels = sound.getnchannels()

        print('Processing file {}: channels = {}, frames count = {}, frame rate = {}, sample width = {}, duration = {:.4f} seconds'.format(file, sound_channels, frames_count, frame_rate, sample_width, float(frames_count) / frame_rate))
        print('Description = {}'.format(description))

        raw_sound = sound.readframes(frames_count)
        time = 0

        spectrogram = numpy.ndarray((rows, max_spectrogram_length))

        index = 0
        while time + sample_size < frames_count and index < max_spectrogram_length:

            raw_bytes = raw_sound[time * 2: (time + sample_size) * 2]
            converted_data = numpy.fromstring(raw_bytes, dtype = numpy.int16)
            fourier = numpy.fft.fft(converted_data)

            # get only half of fourier coefficients
            fourier_normalized_converted = numpy.ndarray((1, rows))
            fourier_normalized_absolute = numpy.ndarray((1, rows))

            minimal_greater_than_zero = float('inf')
            for i in range(8, rows + 8):

                value = numpy.abs(fourier[i])
                if value > 0.0 and minimal_greater_than_zero > value:
                    minimal_greater_than_zero = value
                fourier_normalized_absolute[(0, i - 8)] = value

            # take logarithm and check for infinity
            for i in range(rows):
                fourier_normalized_converted[(0, i)] = 10.0 * numpy.log10(max(fourier_normalized_absolute[(0, i)], minimal_greater_than_zero))

            spectrogram[:, index] = fourier_normalized_converted

            time += time_shift
            index += 1

        # append if necessary
        start_index = 0
        while index < max_spectrogram_length:
            spectrogram[:, index] = spectrogram[:, start_index]
            start_index += 1
            index += 1

        spectrograms.append(spectrogram)
        labels.append(label)

        if file_index == 0:
            figure = matplotlib.pyplot.figure()

        if file_index < display_count:

            subplot = matplotlib.pyplot.subplot(display_count, 1, file_index + 1)
            matplotlib.pyplot.title(description.replace('_', ' '), fontsize=10)

            subplot.set_yticklabels([])
            subplot.set_xticklabels([])

            matplotlib.pyplot.imshow(spectrogram)

        file_index += 1

        if file_index >= dataset_size:
            break

    #matplotlib.pyplot.show()

    print('[' + ctime() + ']: Data preparation is complete.')
    end_time = time_module.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = {} minutes and {} seconds'.format(int(elapsed_time / 60), int(elapsed_time % 60)))

    return (spectrograms, labels)
