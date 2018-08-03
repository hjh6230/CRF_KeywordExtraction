import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Input,Embedding
from keras.layers import LSTM,TimeDistributed,Bidirectional,regularizers
from keras.utils import to_categorical
from word2vec2 import word2vec,loadWordVecs
import nltk
from readin import standardReadin as SR
import numpy as np

num_doc=1880

num_epochs=300

num_steps=4

one_batch_size=40

docInbatch=3

batch_size=one_batch_size*docInbatch  #bumber of total batch


skip_step=4

labelsize=5 # number of label clusters

emb_index=loadWordVecs() #look up table
vocab=len(emb_index)

hidden_size=300

reverse_dict=dict(zip(emb_index.values(), emb_index.keys()))

data = SR("Scopes.xlsx", True)

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
        x = np.zeros((batch_size, num_steps))
        y = np.zeros((batch_size, num_steps, labelsize))
        while True:
            # for idx in range(len(self.data)):
            # mark_idx=self.idx
            for idoc in range(docInbatch):
                text=self.data[self.idx]
                lbs=self.label[self.idx]
                if(len(text)<=skip_step):
                    self.idx = self.idx + 1
                    if (self.idx >= len(self.data)):
                        self.idx = 0
                    continue
                self.current_idx=0
                for i in range(one_batch_size):
                    i_total=i+idoc*one_batch_size
                    if (self.current_idx >= len(text)):
                        self.current_idx = np.random.randint(skip_step)
                    if self.current_idx + num_steps >= len(text):
                        # reset the index back to the start of the data set
                        rest_seat=num_steps-(len(text)-self.current_idx)
                        x[i_total, :] = np.hstack((text[self.current_idx:], np.zeros(rest_seat)))
                        temp_y = np.hstack((lbs[self.current_idx:],np.zeros(rest_seat)))
                        self.current_idx = 0
                    else:
                        x[i_total, :] = text[self.current_idx:self.current_idx + num_steps]
                        temp_y = lbs[self.current_idx:self.current_idx + num_steps]
                    # convert all of temp_y into a one hot representation
                    y[i_total, :, :] = to_categorical(temp_y, num_classes=labelsize)
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
        title = data.getTitle(index+1)
        text = data.getBrief(index+1)
        token_title = nltk.word_tokenize(title)
        token_text = nltk.word_tokenize(text)
        token = token_title
        token.extend(token_text)
        token = [w for w in token if not w in nltk.corpus.stopwords.words('english')]
        token = [w for w in token if w.isalpha()]

        kws = data.loadkw(index+1)
        kws_split = [(word.lower()).split() for word in kws]
        labelList = []
        for tk_O in token:
            lab = 0
            tk = tk_O.lower()
            if tk in nltk.corpus.stopwords.words('english'):
                lab = 4
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
        vecs=[word2vec(emb_index,tk) for tk in token]
        # vecs=np.zeros(len(token))
        # for i in range(len(token)):
        #     # print(token[i])
        #     # ans=word2vec(emb_index, token[i])
        #     vecs[i] = word2vec(emb_index, token[i])
        x.append(np.array(vecs))
        y.append(np.array(labelList))
    size_of_data=len(x)
    bound=int(size_of_data/10*8)

    x_train=np.array(x)[:bound]
    x_valid=np.array(x)[bound:]

    y_train =np.array(y)[:bound]
    y_valid =np.array(y)[bound:]


    return  x_train,y_train,x_valid,y_valid



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy =[]
        self.val_loss = []
        self.val_acc = []
        self.best=0


    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('categorical_accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        if (logs.get('val_categorical_accuracy')>self.val_acc[self.best]):
            self.best=len(self.val_acc)-1

        if (len(self.accuracy)%10==0):
            self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy, 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses, 'g', label='train loss')

        # plt.plot(iters, self.val_acc, 'b', label='val acc')
        # val_loss
        plt.plot(iters, self.val_loss, 'k', label='val loss')

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        # plt.ylim((0,2))
        # plt.show()
        filename='pic/lstm_'+str(num_doc)+'_'+str(num_epochs)+'_'+str(len(self.losses))+'.png'
        plt.savefig(filename)
        print('best model is',self.best)
        print ('val-acc is',self.val_acc[self.best])



def defLSTM():
    model = Sequential()
    model.add(Embedding(vocab+1, hidden_size, input_length=num_steps, mask_zero=True,dropout=0.5))
    # model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True),input_shape=(num_steps,hidden_size),merge_mode='concat'))
    # model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(labelsize)))
    model.add(Activation('softmax'))



    return model

def trainLSTM(model,train_data_generator,valid_data_generator):

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(filepath='dLSTMmodels/500model-'+str(num_doc)+'-{epoch:02d}.hdf5', verbose=1,save_best_only=True)

    history = LossHistory()

    pout=keras.callbacks.History()

    model.fit_generator(train_data_generator.generate(), (num_doc*8)// (10*docInbatch), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=(num_doc*2)// (10*docInbatch), callbacks=[history,checkpointer,pout])


    history.loss_plot()



if __name__ == "__main__":
    print("do")
    x_train, y_train, x_valid, y_valid=load_data()
    train_data_generator = KerasBatchGenerator(x_train,y_train)
    valid_data_generator = KerasBatchGenerator(x_valid,y_valid)
    # next(train_data_generator.generate())
    # next(train_data_generator.generate())
    # print(next(train_data_generator.generate()))

    model=defLSTM()
    trainLSTM(model,train_data_generator,valid_data_generator)