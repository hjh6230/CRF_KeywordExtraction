from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import sys, time

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from featureExtraction import featureExtraction as FeatureExyract
from ProcessBar import progressbar

def CRFtrain(No):
    print("extract data")
    X_train, y_train, X_test, y_test = extractdata()
    print("model fitting")


    crf = Cv3(X_train,y_train)
    y_pred = crf.predict(X_test)

    labels = list(crf.classes_)
    labels.remove('None')
    labels.remove('STOP')

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

    filename="CRFmodel"+str(No)+".sav"
    joblib.dump(crf,filename)
    return 0

def extractdata():
    data = FeatureExyract(True)
    datasize = data.getsize()
    bound = int(datasize * 3 / 4)

    bound_train = 20;
    bound_test = 25;

    print("get train set")

    X_train = []
    y_train = []

    process_bar = progressbar(bound_train - 1, '*')

    for index in range(1, bound_train):
        process_bar.progress(index)
        X_train.append(data.getFeatures(index))
        y_train.append(data.getLabel(index))

    print("get test set")

    X_test = []
    y_test = []

    process_bar2 = progressbar(bound_test - bound_train, '*')

    for index in range(bound_train, bound_test):
        process_bar2.progress(index - bound_train)
        X_test.append(data.getFeatures(index))
        y_test.append(data.getLabel(index))

    print('xsize ', len(X_train))
    print('ysize', len(y_train))
    return X_train, y_train, X_test, y_test

def Cv3(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted')
    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=4,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    # labels = list(crf.classes_)
    # labels.remove('O')

    print('best params:', rs.best_params_)
    return rs.best_estimator_



CRFtrain(1)
