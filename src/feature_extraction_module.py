from abc import ABCMeta, abstractmethod

import numpy as np
import peakutils
from scikits.talkbox.features import mfcc


class FeatureExtractorModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, data, params=None):
        pass

    def check(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("input is not array")

    def normalize(self, result, data):
        return result / float(len(data))


class SpectralFeature:
    __metaclass__ = ABCMeta


class TimeFeature:
    __metaclass__ = ABCMeta


class ZeroCrossingRate(FeatureExtractorModel, TimeFeature):
    def get(self, data, params=None):
        self.check(data)
        return self.normalize(((data[:-1] * data[1:]) < 0).sum(), data)


class Energy(FeatureExtractorModel, TimeFeature):
    def get(self, data, params=None):
        self.check(data)
        return self.normalize(np.sum(np.power(data, 2)), data)


class Autocorrelation(FeatureExtractorModel, TimeFeature):
    def get(self, data, params=None):
        self.check(data)
        return self.normalize((data[:-1] * data[1:]).sum(), data)


class SpectralCentroid(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        self.check(data)
        data = np.abs(data)
        return (data * np.arange(len(data))).sum() / float(data.sum())


class SpectralSmoothness(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        self.check(data)
        data = np.abs(data)
        data = 20 * np.log(data)
        return (2 * data[1:-1] - data[:-2] - data[2:]).sum() / 3


class SpectralSpread(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        data = np.abs(data)
        spectral_centroid = params[0]
        return np.sqrt((np.power(np.arange(len(data)) - spectral_centroid, 2) * data).sum() / data.sum())


class SpectralDissymmetry(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        data = np.abs(data)
        spectral_centroid = params[0]
        return np.sqrt(np.abs((np.power(np.arange(len(data)) - spectral_centroid, 3) * data).sum() / data.sum()))


class LinearRegression(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        Nb = len(data)
        F = np.arange(Nb)
        a = data
        beta = (Nb * (F * a).sum() - F.sum() * a.sum()) / (Nb * np.power(F, 2).sum() - np.power(F, 2).sum())
        return beta


class Rolloff(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        partial_sum = 0.85 * data.sum()
        accumulator = 0.0
        R = 0
        for i in range(len(data)):
            accumulator += data[i]
            if accumulator >= partial_sum:
                R = i
                break
        return R


class SFM(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        accumulator = 0.0
        for i in data:
            accumulator *= i
        accumulator = pow(accumulator, 1.1 / len(data))
        return accumulator / len(data) / data.sum()


class SCF(FeatureExtractorModel, SpectralFeature):
    def get(self, data, params=None):
        return np.max(data) / len(data) / data.sum()


class TrackModel:
    label = ''
    timing_feature = []
    spectral_feature = []
    percussion_feature = []
    mfcc_feature = []

    def __init__(self, label, timing_feature, spectral_feature, percussion_feature, mfcc_feature):
        self.label = label
        self.timing_feature = timing_feature
        self.spectral_feature = spectral_feature
        self.percussion_feature = percussion_feature
        self.mfcc_feature = mfcc_feature

    def to_vector(self):
        return np.concatenate((
            self.timing_feature,
            self.spectral_feature,
            self.percussion_feature,
            self.mfcc_feature
        ))


class FeatureExtractorModule:
    time_feature_models = {}
    spectre_feature_models = {}
    results = {}
    nceps = 0
    frame_size = 512

    def extract_feature(self, track):
        # type :(SpectralTrack)
        timing_feature = self.extract_timing_feature(track)
        spectral_feature = self.extract_spectral_feature(track)
        percussion_feature = self.extract_percussion_feature(track)
        mfcc_feature = self.extract_mfcc(track)
        return TrackModel(track.label,
                          timing_feature=timing_feature,
                          spectral_feature=spectral_feature,
                          percussion_feature=percussion_feature,
                          mfcc_feature=mfcc_feature)

    def extract_percussion_feature(self, track):

        indexes = peakutils.indexes(track.percussion_data,
                                    thres=0.2 / max(track.percussion_data),
                                    min_dist=10)
        period0 = indexes[0]
        amplitude0 = track.percussion_data[indexes[0]]
        if len(indexes) > 2:
            ratioPeriod1 = indexes[1] / indexes[0]
            amplitude1 = track.percussion_data[indexes[1]]
        else:
            ratioPeriod1 = 0
            amplitude1 = 0
        if len(indexes) > 3:
            ratioPeriod2 = indexes[2] / indexes[1]
            amplitude2 = track.percussion_data[indexes[2]]
        else:
            ratioPeriod2 = 0
            amplitude2 = 0
        if len(indexes) > 4:
            ratioPeriod3 = indexes[3] / indexes[2]
            amplitude3 = track.percussion_data[indexes[3]]
        else:
            ratioPeriod3 = 0
            amplitude3 = 0
        return np.array([period0,
                         amplitude0,
                         ratioPeriod1,
                         amplitude1,
                         ratioPeriod2,
                         amplitude2,
                         ratioPeriod3,
                         amplitude3])

    def extract_mfcc(self, track):
        ceps, mspec, spec = mfcc(track.data, nwin=512, nfft=512, nceps=self.nceps)
        return np.nan_to_num(ceps)

    def extract_spectral_feature(self, track):
        result = []
        for i in track.spectral_data:
            self.eval_models(self.spectre_feature_models, i)
            keys = filter(lambda x: issubclass(x, SpectralFeature), self.results)
            values = [self.results[key] for key in keys]
            result.append(values)
        return np.array(result)

    def extract_timing_feature(self, track):
        n_fragments = int(len(track.data) / self.frame_size)
        temp = np.split(track.data[:n_fragments * self.frame_size], n_fragments)
        result = []
        for i in temp:
            self.eval_models(self.time_feature_models, i)
            keys = filter(lambda x: issubclass(x, TimeFeature), self.results)
            result.append(np.array([self.results[key] for key in keys]))
        return np.array(result)

    def eval_models(self, extractors, data):
        for i in range(np.amax(map(lambda x: len(x), extractors.values())) + 1):
            for feature_extractor in filter(lambda x: len(extractors[x]) == i, extractors):
                self.results[feature_extractor] = \
                    feature_extractor() \
                        .get(data,
                             map(lambda x: self.results[x] if x in self.results else None,
                                 extractors[feature_extractor]))

    def __init__(self, models, nceps=16):
        self.nceps = nceps
        time_feature_model_keys = filter(lambda x: isinstance(x(), TimeFeature), models)
        time_feature_model_values = [models[i] for i in time_feature_model_keys]
        for k, v in zip(time_feature_model_keys, time_feature_model_values):
            self.time_feature_models[k] = v
        self.spectre_feature_models = models
        for i in time_feature_model_keys:
            del self.spectre_feature_models[i]
