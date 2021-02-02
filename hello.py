from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from codecs import open
import numpy as np
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

if __name__ == "__main__":

    all_docs, all_labels = read_documents("dataset.txt")

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]  
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    from collections import Counter
    freqsPos = Counter()
    freqsNeg = Counter()

    import itertools
    poSize = 0
    negSize = 0
    for (doc,label) in zip(train_docs,train_labels):
        for w in doc:
            if label == "pos":
                freqsPos[w] += 1
                poSize +=1
            else:
                freqsNeg[w] += 1
                negSize += 1
        
    print(poSize)
    print(negSize)


