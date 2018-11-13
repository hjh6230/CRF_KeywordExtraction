from readin import standardReadin as SR
import numpy as np
import nltk
import codecs

def loadWordVecs():
    embeddings_index = {}
    wv="glove/glove.6B.50d.txt"
    f = open(wv,'r',encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def sent2vec(embeddings_index,s): # this function creates a normalized vector for the whole sentence
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    words = [w for w in words if not w in nltk.corpus.stopwords.words('english')]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())


def word2vec(embeddings_index,w): # this function creates a normalized vector for the whole sentence
    word = str(w).lower()
    # words = [w for w in words if not w in nltk.corpus.stopwords.words('english')]
    # words = [w for w in words if w.isalpha()]
    try:
        M=embeddings_index[word]
    except:
        M= np.zeros(50)

    v = np.array(M)
    if type(v) != np.ndarray:
        return np.zeros(50)
    return v


def vec2word(embeddings_index,v):
    correct_vec=0
    word=' '

    for wd,vec in embeddings_index.items():
        out=np.sum(vec - v)

        if np.abs(out)<0.00000000001:
            correct_vec=1
            word=wd
            break

    return  word




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
    vex=word2vec(emb_index,'apple')
    print(vex)
    wd=vec2word(emb_index,vex)
    print(wd)

