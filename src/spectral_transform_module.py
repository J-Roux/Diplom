import numpy as np
import pywt
import scipy
from scipy import signal
from sklearn.preprocessing import scale

from preprocessing_module import LowPassSinglePole
from track import Track


class SpectralTrack(Track):
    spectral_data = []
    percussion_data = []

    def __init__(self, track, spectral_data, percussion_data):
        self.sample_rate = track.sample_rate
        self.label = track.label
        self.data = track.data
        self.spectral_data = spectral_data
        self.percussion_data = percussion_data


class SpectralTransformer:
    window = signal.hamming(1024)
    level = 4
    alpha = 0.99
    rate = 16

    def __init__(self, window, level, alpha, rate):
        self.window = window
        self.level = level
        self.alpha = alpha
        self.rate = rate

    def short_time_fourier(self, track):
        # type : (Track, list) -> ndarray
        f, t, Zxx = signal.stft(track.data,
                                window=self.window,
                                nperseg=len(self.window))
        return np.abs(Zxx)


    def wavelet_daubechies(self, data):
        data = np.array(pywt.swt(data, 'db4', level=self.level))
        data = np.array([np.sqrt(np.power(i[0], 2) + np.power(i[1], 2)) for i in data])
        data = data.reshape(self.level, data.shape[-1])
        return data

    def __round_to_power_of_two(self, data):
        size = len(data)
        new_size = 2 ** (size.bit_length() - 1)
        return data[:new_size]

    def filter(self, data):
        fltr = LowPassSinglePole(self.alpha)
        result = []
        for i in data:
            result.append(fltr.filter(i))
        return np.array(result)

    def resampling(self, data):
        return data[::self.rate]

    def normalize_and_sum(self, data):
        data = np.array(data)
        accumulator = np.zeros(data.shape[1])
        for i in data:
            accumulator += scale(i, with_mean=True, with_std=False)
        return accumulator

    def autocorrelation(self, data):
        data = scipy.fft(data)
        data = np.abs(scipy.ifft(data * data)) / len(data) / self.level
        return data

    def percussion_correlogramm(self, track):
        data = self.__round_to_power_of_two(track.data)
        data = self.wavelet_daubechies(data)
        results = []
        for i in data:
            filtered = self.filter(i)
            resampled = self.resampling(filtered)
            results.append(resampled)
        data = self.normalize_and_sum(results)
        return data

