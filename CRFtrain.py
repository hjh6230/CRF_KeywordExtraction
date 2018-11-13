from itertools import chain

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV


import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random

from featureExtraction_empty import featureExtraction as FeatureExyract
from ProcessBar import progressbar
import pickle


def CRFvalidation( X_test, y_test,crf):
    # print("extract data")
    # X_train, y_train, X_test, y_test = extractdata()
    print("model fitting")


    # crf = Cv3(X_train,y_train)
    y_pred = crf.predict(X_test)

    labels = list(crf.classes_)
    # labels.remove('None')
    # labels.remove('STOP')

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))




    return 0

def extractdata(selectSize):
    data = FeatureExyract(True)
    datasize = data.getsize()

    name = 'randlist'
    with open('obj/' + name + '.pkl', 'rb') as f:
        randorder = pickle.load(f)
    randorder=randorder[:selectSize]

    X_all=[]
    y_all=[]


    process_bar = progressbar(selectSize, '*')
    count=0

    for index in randorder:
        count=count+1
        process_bar.progress(count)
        X_all.append(data.getFeatures(index))
        y_all.append(data.getLabel(index))



    # shuffle the dataset


    bound = int(selectSize * 3 / 4)
    X_train = X_all[:bound]
    y_train = y_all[:bound]

    X_test = X_all[bound:]
    y_test = y_all[bound:]


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

if __name__ == "__main__":
    for i in range(1):
        datasize=1880
        X_train, y_train, X_test, y_test=extractdata(datasize)
        crf = Cv3(X_train,y_train)
        CRFvalidation(X_train, y_train, crf)
        CRFvalidation(X_test, y_test,crf)
        filename = "models\CRFmodel_empty_"+str(datasize)+"data_"+ str(i) + ".sav"
        joblib.dump(crf, filename)
