from sklearn.naive_bayes import GaussianNB
from codecs import open
import numpy as np
#from __future__ import division

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

if __name__ == "__main__":

    all_docs, all_labels = read_documents("dataset.txt")

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]  
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]


