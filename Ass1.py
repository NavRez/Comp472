from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from codecs import open
import numpy as np
import math
#from __future__ import division

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            new_words = list()
            for word in words[3:]:
                new_words.append(str(word))
            docs.append(new_words)
            labels.append(words[1])
    return docs, labels

def logarithm(PosDict,NegDict,pos,neg,pWords,nWords,train):
    totalpos = math.log(pos/(pos+neg))
    totalneg = math.log(neg/(pos+neg))
    for word in train:
        pval = PosDict[word]+2
        nval = NegDict[word]+2
        totalpos += math.log(pval/(pWords+len(PosDict)*2))
        totalneg += math.log(nval/(nWords+len(NegDict)*2))

    if totalpos > totalneg:
        return "pos"
    else:
        return "neg"




if __name__ == "__main__":

    all_docs, all_labels = read_documents("dataset.txt")

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]  
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    from collections import Counter
    freqsPosWords = Counter()
    freqsNegWords = Counter()

    import itertools
    poSize = 0
    negSize = 0
    posWords = 0
    negWords = 0
    
    for (doc,label) in zip(train_docs,train_labels):
        if label == "pos":
            for w in doc:
                freqsPosWords[w] += 1
                posWords+=1
            poSize +=1
        else:
            for w in doc:
                freqsNegWords[w] += 1
                negWords+=1
            negSize +=1

    print(len(freqsPosWords))
    print(len(freqsNegWords))

    comp_List = list()

    for truedoc in eval_docs:
        comp_List.append(logarithm(freqsPosWords,freqsNegWords,poSize,negSize,posWords,negWords,truedoc))

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    print(accuracy_score(eval_labels,comp_List))
    #print(precision_score(eval_labels,comp_List,average="samples"))
    print(confusion_matrix(eval_labels,comp_List,labels=["pos","neg"]))
    


