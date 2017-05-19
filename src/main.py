import glob
import itertools
import multiprocessing
import os

import numpy as np
import scipy

CPU_COUNT = multiprocessing.cpu_count()

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from matplotlib import pyplot as plt

import unittest
from abc import ABCMeta, abstractmethod


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


class TestFeatureExtractor(unittest.TestCase):
    test_data = np.linspace(-np.pi * 100, np.pi * 100, 500)
    test_data_spectre = scipy.fft(test_data)

    def test_energy(self):
        energy = Energy()
        seq = np.random.random_integers(-573, 573, (1000, 3))
        self.assertGreater(energy.get(seq), 0)

    def test_zero_crossing_rate(self):
        zcr = ZeroCrossingRate()
        self.assertGreater(zcr.get(np.sin(self.test_data)), 200.0 / len(self.test_data))

    def test_first_order_autocorrelation(self):
        autocor = Autocorrelation()
        seq = np.arange(50)
        self.assertEqual(autocor.get(seq), 784.0)

    def test_spectal_centroid(self):
        spcentroid = SpectralCentroid()
        self.assertAlmostEqual(spcentroid.get(self.test_data_spectre), 250.0)

    def test_spectral_smoothness(self):
        spsmoothness = SpectralSmoothness()
        self.assertAlmostEqual(spsmoothness.get(self.test_data_spectre), 235.2160775)

    def test_spectral_spread(self):
        spspread = SpectralSpread()
        spcentroid = SpectralCentroid()
        a = spcentroid.get(self.test_data_spectre)
        spspread.get(self.test_data_spectre, [a])


class FeatureExtractor:
    time_feature_models = {}
    spectre_feature_models = {}
    results = {}

    def set_models(self, models):
        time_feature_model_keys = filter(lambda x: isinstance(x(), TimeFeature), models)
        time_feature_model_values = [models[i] for i in time_feature_model_keys]
        for k, v in zip(time_feature_model_keys, time_feature_model_values):
            self.time_feature_models[k] = v
        self.spectre_feature_models = models
        for i in time_feature_model_keys:
            del self.spectre_feature_models[i]

    def eval_models(self, extractors, data):
        for i in range(np.amax(map(lambda x: len(x), extractors.values())) + 1):
            for feature_extractor in filter(lambda x: len(extractors[x]) == i, extractors):
                self.results[feature_extractor] = \
                    feature_extractor().get(data,
                                            map(lambda x: self.results[x] if x in self.results else None,
                                                extractors[feature_extractor]))

    def get(self, data, fs, frame_size_sec):
        data = scale(data)
        size = int(len(data) / fs / frame_size_sec)
        frames = np.split(data[:int(size * fs * frame_size_sec)], size)
        results = []
        for frame in frames:
            self.eval_models(self.time_feature_models, frame)
            spectre_feature = np.abs(scipy.fft(frame))
            self.eval_models(self.spectre_feature_models, spectre_feature)
            results.append(self.results.values())
        return np.array(results)


models = {
    Energy: [],
    ZeroCrossingRate: [],
    Autocorrelation: [],
    SpectralCentroid: [],
    SpectralSmoothness: [],
    SpectralSpread: [SpectralCentroid],
    SpectralDissymmetry: [SpectralCentroid],
    Rolloff: [],
    LinearRegression: [],
    SFM: [],
    SCF: []
}

genre_list = ['classical',
              'jazz',
              'country',
              'pop',
              'rock',
              'metal',
              'blues',
              'disco',
              'hiphop',
              'reggae']

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, activation='tanh'),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

from scikits.talkbox.features import mfcc


def read_feature(genre_list, base_dir):
    X = []
    Y = []
    for label, genre in enumerate(genre_list):
        print genre
        genre_dir = os.path.join(base_dir, genre, "*.fft")
        for fn in glob.glob(genre_dir):
            fft_features = np.loadtxt(fn)
            fft_features = np.nan_to_num(fft_features)
            len_fft = len(fft_features)
            mean = np.mean(fft_features[int(len_fft * 0.2):int(len_fft * 0.8)], axis=0)
            std = np.std(fft_features[int(len_fft * 0.2):int(len_fft * 0.8)], axis=0)
            X.append(np.concatenate((mean, std)))
            Y.append(label)
    return np.array(X), np.array(Y)


def classify(data, labels, name, clf):
    predicted = cross_val_predict(clf, data, labels, cv=10)
    cnf_matrix = confusion_matrix(labels, predicted)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=genre_list, normalize=True,
                          title=name)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()








import peakutils


def get_percusion_data(frame):
    data = np.array(pywt.swt(frame, 'db4', level=4))
    data = np.array([np.sqrt(np.power(i[0], 2) + np.power(i[1], 2)) for i in data])
    data = data.reshape(4, data.shape[-1])
    decay = 0.99
    results = []
    for i in data:
        fltr = LowPassSinglePole(decay)
        result = []
        for j in i:
            result.append(fltr.filter(j))
        results.append(result[::16])
    data = np.array(results)
    data = scale(data[0], with_mean=True) + scale(data[1], with_mean=True) + \
           scale(data[2], with_mean=True) + scale(data[3], with_mean=True)

    data = scipy.fft(data)
    data = np.abs(scipy.ifft(data * data)) / len(data) / 4
    indexes = peakutils.indexes(data, thres=0.2 / max(data), min_dist=10)
    period0 = indexes[0]
    amplitude0 = data[indexes[0]]
    if (len(indexes) > 2):
        ratioPeriod1 = indexes[1] / indexes[0]
        amplitude1 = data[indexes[1]]
    else:
        ratioPeriod1 = 0
        amplitude1 = 0
    if (len(indexes) > 3):
        ratioPeriod2 = indexes[2] / indexes[1]
        amplitude2 = data[indexes[2]]
    else:
        ratioPeriod2 = 0
        amplitude2 = 0
    if (len(indexes) > 4):
        ratioPeriod3 = indexes[3] / indexes[2]
        amplitude3 = data[indexes[3]]
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


def create_feature(fn):
    sample_rate, X = read(fn)
    feature_extractor = FeatureExtractor()
    feature_extractor.set_models(models)
    fft_features = feature_extractor.get(X, sample_rate, 1)
    size = int(len(X) / sample_rate / 1)
    frames = np.split(X[:int(size * sample_rate * 1)], size)
    percusion_feature = []
    cepstral_feature = []
    for i in frames:
        percusion_feature.append(get_percusion_data(i[:int(16384 * 1)]))
        cepstral_feature.append(np.mean(np.nan_to_num(mfcc(i, nwin=512, nfft=512, fs=sample_rate)[0]), axis=0))
    result = np.concatenate((fft_features, percusion_feature, cepstral_feature), axis=1)
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.savetxt(data_fn, np.array(result))
    print fn





from scipy import signal

from read_wav_module import WavModule
from preprocessing_module import PreprocessingModule
from spectral_transform import SpectralTransformer
import platform

path = ''
path_to_wav = ''

if platform.system() == 'Windows':
    path = 'C:\\Users\\Pavel\\Downloads\\genres'
    path_to_wav = path + '\\*\\*.wav'
else:
    path = '/home/pavel/Downloads/genres'
    path_to_wav = path + '/*/*.wav'

if __name__ == '__main__':
    plt.interactive(False)
    np.set_printoptions(precision=10)
    file_list = glob.glob(path + '/*/*.au')
    # Parallel(n_jobs=CPU_COUNT)(
    #   delayed(create_feature)(wav_file) for wav_file in file_list
    # )
    # print 'create feature -- done'

    # data, labels = read_feature(genre_list, path)
    # print data.shape
    # print 'read feature -- done'
    # data = np.nan_to_num(data)
    # data = scale(data)
    # data = np.nan_to_num(data)
    # print np.isinf(data).any()
    # print np.isnan(data).any()
    read_wav_module = WavModule()
    preprocessing_module = PreprocessingModule(alpha=0.99, cut_end=0.1, cut_start=0.1, overlap=0.5)
    spectral_transformer = SpectralTransformer(decay=0.99, level=4, rate=16, window=signal.hamming(1024))
    print file_list[0]
    track = read_wav_module.read_wav(file_list[0].replace("au", 'wav'), 'blues')
    track = preprocessing_module.cutting(track)
    track = preprocessing_module.scale(track)
    track = preprocessing_module.stereo_to_mono(track)
    # track = preprocessing_module.filter(track)
    tracks = preprocessing_module.framing(track, 5)
    spectral = spectral_transformer.short_time_fourier(track)
    percussion = spectral_transformer.percussion_correlogramm(track)
    plt.plot(percussion)
    plt.show()
    # Parallel(n_jobs=CPU_COUNT)(
    #    delayed(classify)(data, labels, name, clf) for name, clf in zip(names, classifiers)
    # )
