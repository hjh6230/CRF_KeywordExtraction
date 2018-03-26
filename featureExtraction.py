import numpy as np
import nltk
from readin import standardReadin as SR




# def sent2features(sent):
#     return [word2features(sent, i) for i in range(len(sent))]
#
# def sent2labels(sent):
#     return [label for token, postag, label in sent]
#
# def sent2tokens(sent):
#     return [token for token, postag, label in sent]

class featureExtraction:
    def __init__(self, islabeled=False, filename="Scopes.xlsx"):
        self.data=SR(filename,islabeled)
        self.datasize=self.data.getsize()
        self.isLabeled=islabeled

    def getsize(self):
        return self.datasize

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
            feature_i['inTitle']=True
            objList.append(feature_i)
        for i in range(len(token_text)):
            feature_i=self.word2features(token_text,i)
            feature_i['inTitle']=False
            objList.append(feature_i)
        return objList


    def getLabel(self,index):
        if not self.isLabeled:
            return 0
        title = self.data.getTitle(index)
        text = self.data.getBrief(index)
        token= nltk.word_tokenize(title)
        token_text = nltk.word_tokenize(text)
        token.extend(token_text)
        #print(token)
        kws=self.data.loadkw(index)
        kws_split=[(word.lower()).split() for word in kws]
        labelList=[]
        for tk_O in token:
            tk=tk_O.lower()
            lab='None'
            for kw in kws_split:
                if (tk in kw):
                    if len(kw) == 1:
                        lab = 'KW_A'
                        break
                    if kw.index(tk) == 0: lab = 'KW_S'
                    else: lab = 'KW_M'
                    break
            if (lab == 'None'):
                if (tk in self.data.stoplist):lab='STOP'
            labelList.append(lab)
        return labelList







        # get Pos



    def getPos(self,token):
        result=nltk.pos_tag(token)
        #print (result)
        pair=result[0]
        tag=pair[1]
        # question:how to represent tag
        return tag

    def getChunk(self,tagged):
        chunktags = []
        chunkGram = '''
                    NP: {<DT>? <JJ>* <NN>*}
                    P: {<IN>}
                    V: {<V.*>}
                    RVN:{<RB.?>*<VB.?>*<NNP>+<NN>?}'''
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)

        # print(chunked)
        # for subtree in chunked.subtrees():
        #     print( subtree.label())
        # print (chunked.leaves())
        for i in range(len(chunked)):
            if (type(chunked[i]) == nltk.tree.Tree):
                for tup in chunked[i]:
                    chunkTag = (tup[0], chunked[i].label())
                    chunktags.append(chunkTag)
            else:
                chunkTag = (chunked[i][0], 'None')
                chunktags.append(chunkTag)


        return (chunktags)

    def getNgram(self,token):
        return 0

    def word2features(self,sent, i):

        posList = nltk.pos_tag(sent)
        chunkList=self.getChunk(posList)



        word = sent[i]
        postag=posList[i][1]
        chunktag=chunkList[i][1]
        freq_title = self.token_title_l.count(word.lower())/len(self.token_title_l)
        if(len(self.token_text_l)==0):
            freq_text=0
        else:
            freq_text = self.token_text_l.count(word.lower())/len(self.token_text_l)
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'freq_title': freq_title,
            'freq_text': freq_text,
            'postag': postag,
            'postag[:2]': postag[:2],
            'chunktag': chunktag,
        }
        if i > 0:
            word1 = sent[i - 1]
            postag1 = posList[i-1][1]
            chunktag1 = chunkList[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
                '-1:chunktag': chunktag1,
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            postag1 = posList[i+1][1]
            chunktag1 = chunkList[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
                '+1:chunktag': chunktag1,
            })
        else:
            features['EOS'] = True

        return features


# fe=featureExtraction(True)
# index=2
# features=fe.getFeatures(2)
# labels=fe.getLabel(2)
# print(len(features),' Vs ',len(labels))



