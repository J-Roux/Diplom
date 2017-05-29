import glob
import multiprocessing
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from genre_classification_module import GenreClassificationModule
from music_feature_extractor import MainModule
from DatabaseClient import DatabaseModule
from visualizer_data_module import VisualizeDataModule
import pickle
import itertools
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

db = DatabaseModule('localhost', 27017)
def extract_and_save():
    X = []
    Y = []
    mfe = MainModule()
    meta = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(path, genre, "*.wav")
        for fn in glob.glob(genre_dir)[:1]:
            print fn
            track_models = mfe.get_feature(fn, [label, fn])
            for i in track_models:
                db.store(i)
                X.append(i.to_vector())
                Y.append(i.label[0])
                meta.append(i.label[1])
                # np.savetxt(path + '\\result.data', X)
                # np.savetxt(path + '\\label.data', Y)
                # pickle.dump(meta,  open(path + '\\meta.data', 'wb'))



def load_data():
    X = np.loadtxt(path + '\\result.data')
    Y = np.loadtxt(path + '\\label.data')
    X = np.nan_to_num(X)
    X = scale(X, axis=0)
    return X, Y


if __name__ == '__main__':
    visualizer = VisualizeDataModule()
    module = GenreClassificationModule(cv=10, labels_name=genre_list)
    plt.interactive(False)
    np.set_printoptions(precision=10)
    extract_and_save()
    # X, Y = load_data()
    #meta = pickle.load(open(path + '\\meta.data', 'rb'))

    # result = module.classify(X, Y, meta)

    # for i in result:
    #    cm = module.plot_confusion_matrix(result[i][1], i)
    #    print i + ' ' + str(sum(cm[i][i] for i in xrange(len(cm))) / 10)
    #   visualizer.plot_3d(X, labels=Y, genre_list=genre_list[:3], reduction_method='t_sne')
    #   visualizer.plot_2d(X, labels=Y, genre_list=genre_list[:3], reduction_method='t_sne')
