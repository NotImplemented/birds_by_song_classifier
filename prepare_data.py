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


output_classes = 35

sample_size = 512
time_shift = int(sample_size / 2)

max_song_length = 30
sample_length = 5 # sample length in seconds

max_frame_rate = 44100
max_spectrogram_length = int((max_frame_rate * sample_length - sample_size) / time_shift + 1)
rows = int(sample_size / 2)

image_rows = rows
image_columns = max_spectrogram_length
dataset_size = 4

wav_path_train_nips = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', 'train')
wav_path_test_nips = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', 'test')

wav_path_train_mlsp = os.path.join('mlsp_train_set', 'train_set')
wav_path_test_mlsp = os.path.join('mlsp_train_set', 'test_set')

labels_path = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS', 'nips4b_birdchallenge_train_labels.csv')

def cook_spectrogram(file_path):

    _, extension = os.path.splitext(file_path)

    # read raw sound and build spectrogram
    sound = wave.open(file_path, 'r')
    frames_count = sound.getnframes()
    frame_rate = sound.getframerate()
    sample_width = sound.getsampwidth()
    sound_channels = sound.getnchannels()

    print('Processing file {}: channels = {}, frames count = {}, frame rate = {}, sample width = {}, duration = {:.4f} seconds'.format(
            file_path, sound_channels, frames_count, frame_rate, sample_width, float(frames_count) / frame_rate))

    raw_sound = sound.readframes(frames_count)
    time = 0

    spectrogram = numpy.ndarray((rows, int((frames_count - sample_size) / time_shift) + 1 ))

    index = 0
    while time + sample_size < frames_count:

        raw_bytes = raw_sound[time * 2 : (time + sample_size) * 2]
        converted_data = numpy.fromstring(raw_bytes, dtype=numpy.int16)
        fourier = numpy.fft.fft(converted_data)

        # use only half of fourier coefficients
        fourier_normalized = numpy.ndarray((1, rows))
        fourier_absolute = numpy.ndarray((1, rows))

        for i in range(rows):

            value = numpy.abs(fourier[i])
            fourier_absolute[(0, i)] = value

        # take logarithm and check for infinity
        for i in range(rows):
            fourier_normalized[(0, i)] = fourier_absolute[(0, i)]

        mx = numpy.max(fourier_normalized)
        mn = numpy.min(fourier_normalized)

        #if mn != mx:
        #   fourier_normalized = (fourier_normalized - mn) / (mx - mn)

        spectrogram[:, index] = numpy.sqrt(fourier_normalized + 1.0)

        time += time_shift
        index += 1

    # append if necessary
    start_index = 0
    while index < max_spectrogram_length:
        spectrogram[:, index] = spectrogram[:, start_index]
        start_index += 1
        index += 1

    mx = numpy.max(spectrogram)
    mn = numpy.min(spectrogram)

    spectrogram = (spectrogram - mn) / (mx - mn)

    spectrogram = numpy.sqrt(spectrogram)
    #spectrogram = numpy.max(spectrogram, 0.9) / 10.0

    #for i in range(rows):
    #    for j in range(max_spectrogram_length):
    #        if spectrogram[i, j] < 0.5:
    #            spectrogram[i, j] = 0.0;

    print('Test file {}: max = {}, min = {}'.format(file_path, mx, mn))

    #show_spectrogram(spectrogram, 'sample')
    return spectrogram

def show_spectrogram(spectrogram, description):

    display_count = 1
    figure = matplotlib.pyplot.figure()

    subplot = matplotlib.pyplot.subplot(display_count, 1, 1)

    if description is not None:
        matplotlib.pyplot.title(description.replace('_', ' '), fontsize = 10)

    subplot.set_yticklabels([])
    subplot.set_xticklabels([])

    matplotlib.pyplot.imshow(spectrogram)
    matplotlib.pyplot.show()

def prepare_train_dataset():

    print('[' + ctime() + ']: Train data preparation has started.')
    start_time = time_module.time()

    pixel_type = numpy.dtype(numpy.uint16)
    factor = 1.0 / (numpy.iinfo(pixel_type).max - numpy.iinfo(pixel_type).min)

    labels = []
    spectrograms = []

    file_index = 0
    for file in os.listdir(wav_path_train_mlsp):

        # read labels and set up classes
        label = numpy.zeros((1, output_classes))
        label[(0, file_index)] = 1.0

        file_path = os.path.abspath(os.path.join(wav_path_train_mlsp, file))
        spectrogram = cook_spectrogram(file_path)
        if spectrogram is None:
            continue

        _, max_length = spectrogram.shape
        length = 0
        while length + image_columns < max_length:
            sample = spectrogram[:, length : length + image_columns]

            spectrograms.append(sample)
            labels.append(label)

            length += int(image_columns / 2)

        file_index += 1
        if file_index >= dataset_size:
            break

    print('[' + ctime() + ']: Data preparation is complete.')
    end_time = time_module.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = {} minutes and {} seconds'.format(int(elapsed_time / 60), int(elapsed_time % 60)))

    return (spectrograms, labels)

def prepare_test_dataset():

    print('[' + ctime() + ']: Test data preparation has started.')
    start_time = time_module.time()

    files = []
    spectrograms = []

    file_index = 0
    for file in os.listdir(wav_path_test_mlsp):

        label = numpy.zeros((1, output_classes))

        file_path = os.path.abspath(os.path.join(wav_path_test_mlsp, file))
        spectrogram = cook_spectrogram(file_path)
        if spectrogram is None:
            continue

        spectrograms.append(spectrogram)
        files.append(os.path.basename(file_path))

        file_index += 1
        if file_index >= dataset_size:
            break

    print('[' + ctime() + ']: Test data preparation is complete.')
    end_time = time_module.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = {} minutes and {} seconds'.format(int(elapsed_time / 60), int(elapsed_time % 60)))

    return (files, spectrograms)
