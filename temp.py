import keras
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation,Input
from keras.layers import LSTM,TimeDistributed,Bidirectional
from keras.utils import to_categorical
from word2vec2 import word2vec,loadWordVecs,vec2word,featureExtraction
import nltk
from readin import standardReadin as SR
import numpy as np
from ProcessBar import progressbar
import pickle
from sklearn.metrics import precision_score, recall_score

num_doc=300
num_epochs=300

num_steps=8

one_batch_size=20

docInbatch=3

batch_size=one_batch_size*docInbatch  #bumber of total batch


skip_step=4

labelsize=5 # number of label clusters

emb_index=loadWordVecs() #look up table
vocab=len(emb_index)

# hidden_size=100

feature_size=9

reverse_dict=dict(zip(emb_index.values(), emb_index.keys()))

data = SR("Scopes.xlsx", True)

class KerasBatchGenerator(object):

    def __init__(self, datain ,label):
        self.data = datain
        self.datasize=len(datain)
        self.label=label
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        self.idx=0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.labelsize=labelsize
        self.count=0


    def generate(self):
        while True:

            text = self.data[self.idx]
            lbs = self.label[self.idx]

            len_batch = int(len(text) / skip_step)
            if len_batch==0:len_batch=1

            id = np.zeros((len_batch, num_steps))
            x = np.zeros((len_batch, num_steps, feature_size))
            y = np.zeros((len_batch, num_steps, labelsize))

            if (len(text) <= skip_step):
                self.idx = self.idx + 1
                if (self.idx >= len(self.data)):
                    self.idx = 0
                yield [id,x], y
            self.current_idx = 0
            for i in range(len_batch):

                head=0
                if (self.current_idx >= len(text)):
                    head+=1
                    self.current_idx = head
                if self.current_idx + num_steps >= len(text):
                    # reset the index back to the start of the data set
                    rest_seat=num_steps-(len(text)-self.current_idx)
                    id[i, :] = np.hstack((text[self.current_idx:, 0], np.zeros((rest_seat))))
                    x[i, :, :] = np.vstack((text[self.current_idx:,1:], np.zeros((rest_seat,feature_size))))
                    temp_y = np.hstack((lbs[self.current_idx:],np.zeros((rest_seat))))
                    self.current_idx = 0
                else:
                    id[i, :] = text[self.current_idx:self.current_idx + num_steps, 0]
                    x[i, :, :] = text[self.current_idx:self.current_idx + num_steps,1:]
                    temp_y = lbs[self.current_idx:self.current_idx + num_steps]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=labelsize)
                self.current_idx += skip_step

            self.idx=self.idx+1
            if(self.idx>=len(self.data)):
                self.idx=0
            yield [id, x], y

def load_data():
    # get the data paths
    # data = SR("Scopes.xlsx", True)
    x=[]
    y=[]

    datasize = data.getsize()

    # randorder = [i for i in range(1, datasize)]
    # np.random.shuffle(randorder)
    name = 'randlist'
    f=open('obj/' + name + '.pkl', 'rb')
    randorder=pickle.load(f)
    randorder = randorder[1800 - num_doc:1800]
    ft=featureExtraction(emb_index)



    # for index in range(num_doc):
    #     title = data.getTitle(index+1)
    #     text = data.getBrief(index+1)
    #     token_title = nltk.word_tokenize(title)
    #     token_text = nltk.word_tokenize(text)
    #     token = token_title
    #     token.extend(token_text)
    #     token = [w for w in token if not w in nltk.corpus.stopwords.words('english')]
    #     token = [w for w in token if w.isalpha()]
    #
    #     kws = data.loadkw(index+1)
    #     kws_split = [(word.lower()).split() for word in kws]
    #     labelList = []
    #     for tk_O in token:
    #         lab = 0
    #         tk = tk_O.lower()
    #         if tk in nltk.corpus.stopwords.words('english'):
    #             lab = 4
    #         for kw in kws_split:
    #             if (tk in kw):
    #                 if len(kw) == 1:
    #                     lab = 1
    #                     break
    #                 if kw.index(tk) == 0:
    #                     lab = 2
    #                 else:
    #                     lab = 3
    #                 break
    #         labelList.append(lab)
    #     vecs=[word2vec(emb_index,tk) for tk in token]
    #     # vecs=np.zeros(len(token))
    #     # for i in range(len(token)):
    #     #     # print(token[i])
    #     #     # ans=word2vec(emb_index, token[i])
    #     #     vecs[i] = word2vec(emb_index, token[i])
    #     x.append(np.array(vecs))
    #     y.append(np.array(labelList))

    process_bar = progressbar(num_doc, '*')
    count = 0

    for index in randorder:
        count = count + 1
        process_bar.progress(count)
        x.append(np.array(ft.getFeatures(index)))
        y.append(np.array(ft.getLabel(index)))
    size_of_data=len(x)
    x=np.array(x)

    y=np.array(y)


    return  x,y,randorder



if __name__ == "__main__":
    x,y,randorder=load_data()
    modelname="1880model-100-175"
    model= load_model("dLSTMmodels/"+modelname+".hdf5")
    gen=KerasBatchGenerator(x,y)

    kwNum = 0
    predNum = 0.
    correct = 0
    get = 0

    id_doc=0

    Answer=[]
    Pred=[]

    filename = "sample/lstm_" + modelname + "_ext_" + str(num_doc) + '.txt'
    file = open(filename, "w",encoding='utf-8')

    mtx=[[],[]]


    for i in range(num_doc):
        doc_idx = randorder[i]
        kws = data.loadkw(doc_idx)
        # if (i==490):
        #     print(i)

        # start to extract kw from predict
        gen_data=next(gen.generate())
        cont=gen_data[0]

        y_val=gen_data[1]

        if len(cont)==1:
            print(i)

        pred=model.predict([cont[0],cont[1]])
        cont = cont[0]

        pred_label=np.argmax(pred,axis=2)
        y_val=np.argmax(y_val,axis=2)

        labels=[0,1,2,3,4]

        prec=0
        recall=0
        for j in range(len(y_val)):
            prec+=precision_score(y_val[j], pred_label[j],average='weighted')
            recall+=recall_score(y_val[j], pred_label[j],average='weighted')
        prec/=len(y_val)
        recall/=len(y_val)

        mtx[0].append(prec)
        mtx[1].append(recall)

        pred_kw=[]
        for seq in range(len(pred_label)):
            for word in range(num_steps):
                wd = 0
                if pred_label[seq,word]== 1:
                    wd = vec2word(reverse_dict,cont[seq,word])
                if pred_label[seq,word] == 2:
                    wd = vec2word(reverse_dict,cont[seq,word])
                    step = 1
                    while (True):
                        if (word + step >= num_steps):
                            break
                        if pred_label[seq,word+step] != 3:
                            break
                        wd = wd + ' ' + vec2word(reverse_dict,cont[seq,word+step])
                        step = step + 1
                    if(step==1):
                        wd=0

                if (wd in pred_kw) or (wd == 0):
                    continue
                else:
                    pred_kw.append(wd)
        kwNum +=  len(kws)
        predNum += len(pred_kw)

        kws_low = [kw.lower() for kw in kws]



        print("document num:" + str(randorder[i]), file=file)
        print("predictions:", file=file)
        for kw in pred_kw:
            print(kw, file=file)
        print("answers:", file=file)
        for kw in kws_low:
            print(kw, file=file)
        print("         ", file=file)


        for kw in pred_kw:
            if kw in kws_low:
                correct+=1

        for kw in kws_low:
            if kw in pred_kw:
                get+=1
        Answer.append(kws_low)
        Pred.append(pred_kw)

    print("recall = ", np.mean(mtx[1]))
    print("precision=", np.mean(mtx[0]))

    print ('kws')

    print("recall = ", get / kwNum)
    print("precision=", correct / predNum)
