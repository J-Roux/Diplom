import itertools
import multiprocessing

import numpy as np
from joblib import Parallel
from joblib import delayed
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CPU_COUNT = multiprocessing.cpu_count()


def classify_p(clf, clf_name, data, labels, cv):
    predicted = GenreClassificationModule.cross_validation_predict(clf, data, labels, cv=cv)
    scores = GenreClassificationModule.cross_val_score(clf, data, labels, cv=cv)
    accuracy, std = scores.mean(), scores.std()
    cnf_matrix = confusion_matrix(labels, predicted)
    return {clf_name: [(accuracy, std), cnf_matrix]}


clf = {
    'Nearest Neighbors': KNeighborsClassifier(3),
    'Linear SVM': SVC(kernel="linear", C=0.025),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=10, max_features=1),
    'Neural Net': MLPClassifier(hidden_layer_sizes=(1000,), alpha=1),
    'AdaBoost': AdaBoostClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis()
}


class GenreClassificationModule:
    classifiers = {}
    labels_name = []
    cv = 3

    def __init__(self, labels_name, cv, classifiers=clf):
        self.classifiers = classifiers
        self.labels_name = labels_name
        self.cv = cv

    @staticmethod
    def cross_validation_predict(clf, data, labels, cv):
        predicted = cross_val_predict(clf, data, labels, cv=cv)
        return predicted

    @staticmethod
    def cross_val_score(clf, data, labels, cv):
        result = cross_val_score(clf, data, labels, cv=cv)
        return result

    def plot_confusion_matrix(self, cnf_matrix, clf_name):
        self.__plot_confusion_matrix(cnf_matrix,
                                     classes=self.labels_name,
                                     normalize=True,
                                     title=clf_name
                                     )
        plt.show()

    def __plot_confusion_matrix(self, cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
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

    def classify(self, data, labels):
        temp = Parallel(n_jobs=CPU_COUNT)(
            delayed(classify_p)
            (self.classifiers[name], name, data, labels, self.cv) for name in self.classifiers
        )
        result = dict()
        for i in temp:
            result.update(i)
        return result
