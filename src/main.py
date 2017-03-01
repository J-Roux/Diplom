#! /usr/bin/Python
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft as scipy_fft
import matplotlib.pyplot as plt
import sklearn
import pywt
import subprocess
import os
from joblib import Parallel, delayed
import multiprocessing
import time

CPU_COUNT = multiprocessing.cpu_count()


def fft(wav_info):
    (rate, data) = wav_info
    window_size = 512
    new_len =  (len(data) / 512) * 512
    data = data[:new_len]
    windows = np.split(data, new_len / window_size)
    return [scipy_fft(window) for window in windows]


def temp(wav_file):
    images = fft(wavfile.read(wav_file))
    texture_window = np.array_split(images, len(images) / 40)
    print texture_window

if __name__ == '__main__':
    start = time.time()
    plt.interactive(False)
    wav_files = [f for f in os.listdir('./wav')]
    wav_infos = Parallel(n_jobs=CPU_COUNT)(delayed(temp)('./wav/' + wav_file) for wav_file in wav_files)
    end = time.time()
    print (end - start)