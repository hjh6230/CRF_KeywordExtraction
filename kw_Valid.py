
from sklearn.externals import joblib

import random
import CRFtrain

from featureExtraction import featureExtraction as FeatureExyract
from ProcessBar import progressbar

def testmodel ( selectSize ):

    #extract data
    data = FeatureExyract(True)

    datasize = data.getsize()

    randorder = [i for i in range(1,datasize+1)]
    random.shuffle(randorder)
    randorder=randorder[:selectSize]

    X_all=[]
    kw_all=[]


    process_bar = progressbar(selectSize, '*')
    count=0

    for index in randorder:
        count=count+1
        process_bar.progress(count)
        X_all.append(data.getFeatures(index))
        kw_all.append(data.data.loadkw(index))

    # load model

    filename="CRFmodel_500data_0.sav"
    crf= joblib.load(filename)
    y_pred = crf.predict(X_all)
    kwNum = 0
    predNum=0
    correct=0
    miss=0

    process_bar = progressbar(selectSize, '*')



    for i in range(selectSize):
        process_bar.progress(i)
        labels=y_pred[i]
        words=X_all[i]
        kws=kw_all[i]

        pred_kw=[]
        for j in range(len(labels)):
            wd=0
            if labels[j]=='KW_A':
                wd=words[j]["word.lower()"]
            if labels[j]=='KW_S':
                wd = words[j]["word.lower()"]
                step=1
                while(True):
                    if(j+step>=len(labels)):
                        break
                    if labels[j + step] != 'KW_M':
                        break
                    wd=wd+' '+words[j+step]["word.lower()"]
                    step=step+1

            if (wd in pred_kw)or(wd == 0):
                continue
            else:
                pred_kw.append(wd)
        kwNum = kwNum+len(kws)
        predNum = predNum+len(pred_kw)

        kws_low=[kw.lower() for kw in kws]

        # for kw in kws_low:
        #     if not kw in pred_kw:
        #         miss=miss+1
        for pred in pred_kw:
            if pred in kws_low:
                correct=correct+1
    print(" ")
    print("recall = ",correct/kwNum)
    print("precision=",correct/predNum)

    return 0

if __name__ == "__main__":
    testmodel(10)






