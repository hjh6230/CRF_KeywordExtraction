
from sklearn.externals import joblib

import random
import CRFtrain

from featureExtraction_empty import featureExtraction as FeatureExyract
from ProcessBar import progressbar
import pickle

def testmodel ( selectSize ):

    #extract data
    data = FeatureExyract(True)

    datasize = data.getsize()

    name = 'randlist'
    with open('obj/' + name + '.pkl', 'rb') as f:
        randorder = pickle.load(f)
    randorder=randorder[1800-selectSize:1800]

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
    datasize=1880
    for index in range(1):
        filename = "models\CRFmodel_empty_" + str(datasize) + "data_" + str(index) + ".sav"
        crf= joblib.load(filename)
        y_pred = crf.predict(X_all)
        kwNum = 0
        predNum=0.
        correct=0
        get=0

        process_bar = progressbar(selectSize, '*')
        filename="sample/CRF_"+str(datasize)+"ext_"+str(index)+'.txt'
        file=open(filename,"w",encoding='utf-8')
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
                    if(step==1):
                        wd=0

                if (wd in pred_kw)or(wd == 0):
                    continue
                else:
                    pred_kw.append(wd)
            kwNum = kwNum+len(kws)
            predNum = predNum+len(pred_kw)

            kws_low=[kw.lower() for kw in kws]

            print("document num:"+str(randorder[i]),file=file)
            print("predictions:", file=file)
            for kw in pred_kw:
                print(kw,file=file)
            print("answers:", file=file)
            for kw in kws_low:
                print(kw, file=file)
            print("         ", file=file)

            # for kw in kws_low:
            #     if not kw in pred_kw:
            #         miss=miss+1
            for pred in pred_kw:
                if pred in kws_low:
                    correct=correct+1
            for kw in kws_low:
                if kw in pred_kw:
                    get += 1
        print("test "+str(index))
        print("recall = ",get/kwNum)
        print("precision=",correct/predNum)


    return 0

if __name__ == "__main__":
    testmodel(300)






