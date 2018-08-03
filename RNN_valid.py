import keras
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation,Input
from keras.layers import LSTM,TimeDistributed,Bidirectional
from keras.utils import to_categorical
from word2vec import word2vec,loadWordVecs,vec2word
import nltk
from readin import standardReadin as SR
import numpy as np
from numpy import  random
from ProcessBar import progressbar

num_doc=10

num_epochs=10

num_steps=8

one_batch_size=30

docInbatch=3

batch_size=one_batch_size*docInbatch  #bumber of total batch


skip_step=3

labelsize=5 # number of label clusters

emb_index=loadWordVecs() #look up table

hidden_size=300

reverse_dict=dict(zip(emb_index.values(), emb_index.keys()))

data = SR("Scopes.xlsx", True)

datasize=data.getsize()


randorder = [i for i in range(1, datasize + 1)]
random.shuffle(randorder)
randorder = randorder[:num_doc]

class KerasBatchGenerator(object):

    def __init__(self, data ,label):
        self.data = data
        self.datasize=len(data)
        self.label=label
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        self.idx=0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.labelsize=labelsize

    def generate(self):
        while True:

            text = self.data[self.idx]
            lbs = self.label[self.idx]

            len_batch=int(len(text)/skip_step)

            x = np.zeros((len_batch, num_steps, hidden_size))
            y = np.zeros((len_batch, num_steps, labelsize))

            self.current_idx=0

            for i in range(len_batch):
                if (self.current_idx >= len(text)):
                    self.current_idx = np.random.randint(skip_step)
                if self.current_idx + num_steps >= len(text):
                    # reset the index back to the start of the data set
                    rest_seat=num_steps-(len(text)-self.current_idx)
                    x[i, :, :] = np.vstack((text[self.current_idx:, :], np.zeros((rest_seat,hidden_size))))
                    temp_y = np.hstack((lbs[self.current_idx:],np.zeros((rest_seat))))
                    self.current_idx = 0
                else:
                    x[i, :, :] = text[self.current_idx:self.current_idx + num_steps,:]
                    temp_y = lbs[self.current_idx:self.current_idx + num_steps]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=labelsize)
                self.current_idx += skip_step

            self.idx=self.idx+1
            if(self.idx>=len(self.data)):
                self.idx=0
            # if(isInv==1):
            #     x=x[:,-1,:]
            #     y=y[:,-1,:]
            #     self.idx=mark_idx
            yield x, y

def load_data():
    # get the data paths
    # data = SR("Scopes.xlsx", True)
    x=[]
    y=[]



    for index in range(num_doc):
        doc_idx=randorder[index]
        title = data.getTitle( doc_idx)
        text = data.getBrief( doc_idx)
        token_title = nltk.word_tokenize(title)
        token_text = nltk.word_tokenize(text)
        token = token_title
        token.extend(token_text)
        token = [w for w in token if not w in nltk.corpus.stopwords.words('english')]
        token = [w for w in token if w.isalpha()]

        kws = data.loadkw( doc_idx)
        kws_split = [(word.lower()).split() for word in kws]
        labelList = []
        for tk_O in token:
            tk = tk_O.lower()
            lab = 0
            for kw in kws_split:
                if (tk in kw):
                    if len(kw) == 1:
                        lab = 1
                        break
                    if kw.index(tk) == 0:
                        lab = 2
                    else:
                        lab = 3
                    break
            labelList.append(lab)
        vecs=np.zeros((len(token),hidden_size))
        for i in range(len(token)):
            # print(token[i])
            # ans=word2vec(emb_index, token[i])
            vecs[i, :] = word2vec(emb_index, token[i])
        x.append(np.array(vecs))
        y.append(np.array(labelList))
    size_of_data=len(x)
    bound=int(size_of_data/4*3)



    return  x,y



if __name__ == "__main__":
    x,y=load_data()
    model= load_model("dLSTMmodels/model-40.hdf5")
    gen=KerasBatchGenerator(x,y)



    kwNum = 0
    predNum = 0.
    correct = 0
    miss = 0

    id_doc=0

    Answer=[]
    Pred=[]


    for i in range(num_doc):
        doc_idx = randorder[i]
        kws = data.loadkw(doc_idx)

        # start to extract kw from predict
        gen_data=next(gen.generate())
        cont=gen_data[0]
        pred=model.predict(gen_data[0])
        pred_label=np.argmax(pred[:,:,:],axis=2)
        #print (pred_label)
        pred_kw=[]
        for seq in range(len(pred_label)):
            for word in range(num_steps):
                wd = 0
                if pred_label[seq,word]== 1:
                    wd = vec2word(emb_index,cont[seq,word])
                if pred_label[seq,word] == 2:
                    wd = vec2word(emb_index,cont[seq,word])
                    step = 1
                    while (True):
                        if (word + step >= num_steps):
                            break
                        if pred_label[seq,word+step] != 3:
                            break
                        wd = wd + ' ' + vec2word(emb_index,cont[seq,word+step])
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
        for kw in pred_kw:
            if kw in kws_low:
                correct+=1
            else:
                miss +=1

        Answer.append(kws_low)
        Pred.append(pred_kw)

    print("recall = ", correct / kwNum)
    print("precision=", correct / predNum)





