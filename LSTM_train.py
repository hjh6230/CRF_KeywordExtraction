import keras
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Input,Embedding,BatchNormalization
from keras.layers import LSTM,TimeDistributed,Bidirectional,regularizers,SimpleRNN
from keras.utils import to_categorical
from word2vec2 import word2vec,loadWordVecs,featureExtraction
import nltk
from readin import standardReadin as SR
import numpy as np
from ProcessBar import progressbar
import pickle

num_doc=1880
num_epochs=300

num_steps=8

one_batch_size=20

docInbatch=3

batch_size=one_batch_size*docInbatch  #bumber of total batch


skip_step=4

labelsize=5 # number of label clusters

emb_index=loadWordVecs() #look up table
vocab=len(emb_index)

hidden_size=128

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

    def generate_all(self):
        id=np.zeros((batch_size, num_steps))
        x = np.zeros((batch_size, num_steps, feature_size))
        y = np.zeros((batch_size, num_steps, labelsize))
        while True:
            if (self.count >= (num_doc * 2) // (10 * docInbatch)):
                self.idx = 0
                self.count = 0

            self.count += 1
            # for idx in range(len(self.data)):
            for idoc in range(docInbatch):
                text = self.data[self.idx]
                lbs = self.label[self.idx]
                if (len(text) <= skip_step):
                    self.idx = self.idx + 1
                    if (self.idx >= len(self.data)):
                        self.idx = 0
                    continue
                self.current_idx = 0
                for i in range(one_batch_size):
                    head = 0
                    i_total = i + idoc * one_batch_size
                    if (self.current_idx > len(text)):
                        head += 1
                        self.current_idx = head
                    if self.current_idx + num_steps >= len(text):
                        # reset the index back to the start of the data set
                        rest_seat = num_steps - (len(text) - self.current_idx)
                        id[i_total,:]=np.hstack((text[self.current_idx:,0], np.zeros((rest_seat))))
                        x[i_total, :, :] = np.vstack((text[self.current_idx:,1:], np.zeros((rest_seat, feature_size))))
                        temp_y = np.hstack((lbs[self.current_idx:], np.zeros((rest_seat))))
                        self.current_idx = 0
                    else:
                        id[i_total, :]=text[self.current_idx:self.current_idx + num_steps,0]
                        x[i_total, :, :] = text[self.current_idx:self.current_idx + num_steps,1:]
                        temp_y = lbs[self.current_idx:self.current_idx + num_steps]
                    # convert all of temp_y into a one hot representation
                    y[i_total, :, :] = to_categorical(temp_y, num_classes=labelsize)
                    self.current_idx += skip_step

                self.idx = self.idx + 1
                if (self.idx >= len(self.data)):
                    self.idx = 0
            yield [id,x], y

    def generate(self):
        id = np.zeros((batch_size, num_steps))
        x = np.zeros((batch_size, num_steps,feature_size))
        y = np.zeros((batch_size, num_steps, labelsize))
        while True:
            for idoc in range(docInbatch):
                text = self.data[self.idx]
                lbs = self.label[self.idx]
                if (len(text) <= skip_step):
                    self.idx = self.idx + 1
                    if (self.idx >= len(self.data)):
                        self.idx = 0
                    continue
                self.current_idx = 0
                for i in range(one_batch_size):
                    head = 0
                    i_total = i + idoc * one_batch_size
                    if (self.current_idx >= len(text)):
                        head += 1
                        self.current_idx = head
                    if self.current_idx + num_steps > len(text):
                        # reset the index back to the start of the data set
                        rest_seat = num_steps - (len(text) - self.current_idx)
                        # a = text[self.current_idx:, 0]
                        # b = np.zeros((rest_seat))
                        # c=np.hstack((text[self.current_idx:, 0], np.zeros((rest_seat))))
                        id[i_total, :] = np.hstack((text[self.current_idx:, 0], np.zeros((rest_seat))))
                        x[i_total, :, :] = np.vstack((text[self.current_idx:, 1:], np.zeros((rest_seat, feature_size))))
                        temp_y = np.hstack((lbs[self.current_idx:], np.zeros((rest_seat))))
                        self.current_idx = 0
                    else:
                        id[i_total, :] = text[self.current_idx:self.current_idx + num_steps, 0]
                        x[i_total, :, :] = text[self.current_idx:self.current_idx + num_steps, 1:]
                        temp_y = lbs[self.current_idx:self.current_idx + num_steps]
                    # convert all of temp_y into a one hot representation
                    y[i_total, :, :] = to_categorical(temp_y, num_classes=labelsize)
                    self.current_idx += skip_step

                self.idx = self.idx + 1
                if (self.idx >= len(self.data)):
                    self.idx = 0
            yield [id, x], y

def load_data():
    # get the data paths
    # data = SR("Scopes.xlsx", True)
    x=[]
    y=[]

    datasize = data.getsize()

    # randorder = [i for i in range(1, datasize)]
    # np.random.shuffle(randorder)
    name = 'randlist2'
    with open('obj/' + name + '.pkl', 'rb') as f:
        randorder=pickle.load(f)
    randorder = randorder[:num_doc]
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
        plt.plot(iters, self.accuracy, 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses, 'g', label='train loss')

        avgacc = self.val_acc
        for i in range(2, len(avgacc) - 3):
            avgacc[i] = np.mean(self.val_acc[i - 2:i + 2])

        plt.plot(iters, avgacc, 'b', label='val acc')
        # val_loss
        avgloss=self.val_loss
        for i in range(2,len(avgloss)-3):
            avgloss[i]=np.mean(self.val_loss[i-2:i+2])
        plt.plot(iters, avgloss, 'k', label='val loss')

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.ylim((0,2))
        # plt.show()
        filename='pic/rnn_'+str(num_doc)+'_'+str(num_epochs)+'_'+str(len(self.losses))+'.png'
        plt.savefig(filename)
        print('best model is',self.best)
        print ('val-acc is',self.val_acc[self.best])
        plt.close()


def defLSTM():
    # model = Sequential()
    # # model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    # model.add(SimpleRNN(hidden_size, return_sequences=True,input_shape=(num_steps,hidden_size)))
    # model.add(SimpleRNN(hidden_size, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(labelsize)))
    # model.add(Activation('softmax'))

    # x=Input(shape=(num_steps,emb_size),name='input')
    # r1=SimpleRNN(hidden_size, return_sequences=True,activation='elu',activity_regularizer = regularizers.l2(0.0001))(x)
    # r2=SimpleRNN(hidden_size, return_sequences=True,activation='elu',activity_regularizer = regularizers.l2(0.0001))(r1)
    # merge_layer=concatenate([r1, r2])
    # y=Dropout(0.5)(merge_layer)
    # y=TimeDistributed(Dense(labelsize))(y)
    # #  activity_regularizer = regularizers.l2(0.01), kernel_regularizer = regularizers.l1(0.01)
    # y=Activation('softmax')(y)
    # model=Model(inputs=[x],outputs=[y])

    id_in = Input(shape=(num_steps,), name='id_in')
    ft_in = Input(shape=(num_steps, feature_size), name='ft_in')
    x = Embedding(vocab + 1, 8, input_length=num_steps, mask_zero=True)(id_in)
    y = BatchNormalization(axis=2, input_shape=(num_steps, feature_size))(ft_in)
    con = keras.layers.concatenate([x, y])
    # z = Dense(hidden_size, activation='relu')(z)
    r1 = SimpleRNN(hidden_size, return_sequences=True, activation='relu', activity_regularizer=regularizers.l2(0.001),dropout=0.3)(con)
    r2=SimpleRNN(hidden_size, return_sequences=True,activation='relu', activity_regularizer=regularizers.l2(0.001))(r1)
    z=keras.layers.concatenate([r1, r2])
    z = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(z)
    z = Dropout(0.5)(z)
    z = TimeDistributed(Dense(labelsize))(z)
    z = Activation('softmax', name='z')(z)

    model = Model(inputs=[id_in, ft_in], outputs=[z])

    return model

def trainLSTM(model,train_data_generator,valid_data_generator):

    # opt=keras.optimizers.SGD(lr=1e-2, momentum=0.0, decay=1e-4, nesterov=False)
    opt=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])




    checkpointer = keras.callbacks.ModelCheckpoint(filepath='LSTMmodels/model-'+str(num_doc)+'-{epoch:02d}.hdf5', verbose=1,save_best_only=True)

    history = LossHistory()

    pout = keras.callbacks.History()


    model.fit_generator(train_data_generator.generate(), (num_doc*3)// (4*docInbatch), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=num_doc // (4*docInbatch), callbacks=[history,checkpointer,pout])


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
    # trainLSTM(model,train_data_generator,valid_data_generator)

    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath='LSTMmodels/' + str(num_doc) + 'model-' + str(hidden_size) + '-{epoch:02d}.hdf5', verbose=1,
        save_best_only=False)

    history = LossHistory()

    pout = keras.callbacks.History()

    model.fit_generator(train_data_generator.generate(), len(x_train) // docInbatch, num_epochs,
                        validation_data=valid_data_generator.generate_all(),
                        validation_steps=(len(x_valid)) // (docInbatch), callbacks=[history, checkpointer, pout])

    history.loss_plot()