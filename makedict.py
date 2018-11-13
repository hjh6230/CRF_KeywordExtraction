from readin import standardReadin as SR
import numpy as np
from collections import Counter
import nltk
import pickle

def makefull():
    file=SR("Scopes.xlsx",True)
    datasize=file.getsize()
    cnt=Counter()
    for index in range(1,datasize+1):
        title=file.getTitle(index)
        text = file.getBrief(index)
        words=str(title)+' '+str(text)
        words = str(words).lower()
        words = nltk.word_tokenize(words)
        words = [w for w in words if not w in nltk.corpus.stopwords.words('english')]
        words = [w for w in words if w.isalpha()]
        cnt.update(words)

    count_pairs = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    name='dictlist'
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(word_to_id, f, pickle.HIGHEST_PROTOCOL)

def readdict():
    name = 'dictlist'
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def makerand():
    randorder = [i for i in range(1, 1880)]
    np.random.shuffle(randorder)
    name='randlist2'
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(randorder, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    #makefull()
    # vocab=readdict()
    # print('a')
    makerand()