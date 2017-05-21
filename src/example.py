import glob
import multiprocessing
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from genre_classification_module import GenreClassificationModule
from music_feature_extractor import MusicFeatureExtractor
from visualizer_data_module import VisualizeDataModule

CPU_COUNT = multiprocessing.cpu_count()

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

import matplotlib.pyplot as plt
import numpy as np

import platform

path = ''
path_to_wav = ''

if platform.system() == 'Windows':
    path = 'C:\\Users\\Pavel\\Downloads\\genres'
    path_to_wav = path + '\\*\\*.wav'
else:
    path = '/home/pavel/Downloads/genres'
    path_to_wav = path + '/*/*.wav'


def extract_and_save():
    X = []
    Y = []
    mfe = MusicFeatureExtractor()
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(path, genre, "*.wav")
        for fn in glob.glob(genre_dir):
            print fn
            track_models = mfe.get_feature(fn, label)
            for i in track_models:
                X.append(i.to_vector())
                Y.append(i.label)
    np.savetxt(path + '\\result.data', X)
    np.savetxt(path + '\\label.data', Y)


def load_data():
    X = np.loadtxt(path + '\\result.data')
    Y = np.loadtxt(path + '\\label.data')
    X = np.nan_to_num(X)
    X = scale(X)
    return X, Y


if __name__ == '__main__':
    visualizer = VisualizeDataModule()
    module = GenreClassificationModule(cv=10, labels_name=genre_list)
    plt.interactive(False)
    np.set_printoptions(precision=10)
    # extract_and_save()
    X, Y = load_data()
    result = module.classify(X, Y)
    for i in result:
        print i + '  ' + str(result[i][0])
        module.plot_confusion_matrix(result[i][1], i)
    visualizer.plot_3d(X, labels=Y, genre_list=genre_list[:3], reduction_method='t_sne')
    visualizer.plot_2d(X, labels=Y, genre_list=genre_list[:3], reduction_method='t_sne')
