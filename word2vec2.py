
from readin import standardReadin as SR
import numpy as np
import nltk
import codecs
from collections import Counter
import pickle

def loadWordVecs():
    name = 'dictlist'
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def word2vec(embeddings_index,w): # this function creates a normalized vector for the whole sentence
    word = str(w).lower()
    # words = [w for w in words if not w in nltk.corpus.stopwords.words('english')]
    # words = [w for w in words if w.isalpha()]
    vocab=len(embeddings_index)
    try:
        M=embeddings_index[word]
    except:
        M= -1
    return M


def vec2word(rev,v):
    try:
        word=rev[v]
    except:
        word=' '

    return  word


class featureExtraction:
    def __init__(self,indic ,filename="Scopes.xlsx"):
        self.data=SR(filename)
        self.datasize=self.data.getsize()
        self.dict=indic




    def getsize(self):
        return self.datasize

    def gettoken(self,index):
        title = self.data.getTitle(index)
        text = self.data.getBrief(index)
        token_title = nltk.word_tokenize(title)
        token_text = nltk.word_tokenize(text)
        token = token_title
        token.extend(token_text)
        return token


    def getFeatures(self,index):
        title = self.data.getTitle(index)
        text = self.data.getBrief(index)
        token_title = nltk.word_tokenize(title)
        token_text = nltk.word_tokenize(text)

        self.token_title_l = nltk.word_tokenize(title.lower())
        self.token_text_l = nltk.word_tokenize(text.lower())
        # token=token1.extend(token2)
        objList=[]

        # object structure:

        for i in range(len(token_title)):
            feature_i=self.word2features(token_title,i)
            feature_i.append(1)
            objList.append(feature_i)
        for i in range(len(token_text)):
            feature_i=self.word2features(token_text,i)
            feature_i.append(0)
            objList.append(feature_i)
        return objList


    def getLabel(self,index):
        title = self.data.getTitle(index)
        text = self.data.getBrief(index)
        token = nltk.word_tokenize(title)
        token2 = nltk.word_tokenize(text)
        token.extend(token2)
        #print(token)
        kws=self.data.loadkw(index)
        kws_split=[(word.lower()).split() for word in kws]
        labelList=[]
        for tk_O in token:
            tk=tk_O.lower()
            lab=0
            for kw in kws_split:
                if (tk in kw):
                    if len(kw) == 1:
                        lab = 1
                        break
                    if kw.index(tk) == 0: lab = 2
                    else: lab = 3
                    break
            if (lab == 'None'):
                if (tk in self.data.stoplist):lab=4
            labelList.append(lab)
        return labelList


    def word2features(self,sent, i):
        word = sent[i]
        freq_title = self.token_title_l.count(word.lower())/len(self.token_title_l)
        if(len(self.token_text_l)==0):
            freq_text=0
        else:
            freq_text = self.token_text_l.count(word.lower())/len(self.token_text_l)
        features = [
            word2vec(self.dict, word.lower()),
            1.0,
            i,
            len(word),
            word.isupper(),
            word.istitle(),
            word.isdigit(),
            freq_title,
            freq_text,
        ]

        return features


if __name__ == "__main__":
    # file=SR("Scopes.xlsx",True)
    # text=file.getBrief(3)
    emb_index=loadWordVecs()
    # ans=sent2vec(emb_index,text)
    # ans2=word2vec(emb_index,"Elkassst")
    # print (text)
    # print(ans)
    # print(ans2)
    # print (len(ans2))
    # vex=word2vec(emb_index,'apple')
    # print(vex)
    # wd=vec2word(emb_index,vex)


    ft=featureExtraction(emb_index)
    wd=ft.getFeatures(3)
    wd=np.array(wd)
    print(wd)



