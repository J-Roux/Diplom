import numpy as np
from sklearn.preprocessing import scale

from track import Track


class LowPassSinglePole:
    def __init__(self, decay):
        self.b = 1 - decay
        self.y = 0

    def filter(self, x):
        self.y += self.b * (x - self.y)
        return self.y


class PreprocessingModule:
    alpha = 0.0
    overlap = 0.0
    cut_start = 0.0
    cut_end = 0.0

    def __init__(self, alpha, overlap, cut_start, cut_end):
        self.alpha = alpha
        self.overlap = overlap
        self.cut_end = cut_end
        self.cut_start = cut_start

    def scale(self, track):
        # type: (Track) -> Track
        track.data = track.data.astype('float64')
        track.data = scale(track.data, with_std=True, with_mean=True)
        return track

    def stereo_to_mono(self, track):
        # type: (Track) -> Track
        if len(track.data.shape) > 1:
            track.data = np.mean(track.data, axis=0)
        return track

    def filter(self, track):
        # type: (Track) -> Track
        fltr = LowPassSinglePole(self.alpha)
        filter = np.vectorize(lambda x: fltr.filter(x))
        track.data = filter(track.data)
        return track

    def framing(self, track, frame_size_sec):
        # type: (Track, int, float) -> [Track]
        frame_size = frame_size_sec * track.sample_rate
        data = track.data
        results = []
        iteration = int((1 - self.overlap) * frame_size)
        stop = (int(len(data) / iteration) - 1) * iteration
        for i in range(0, stop, iteration):
            results.append(Track((track.sample_rate, data[i:i + frame_size]), track.label))
        return results

    def cutting(self, track):
        length = len(track.data)
        track.data = track.data[int(length * self.cut_start)
        : int(length * (1 - self.cut_end))]
        return track
