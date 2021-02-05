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

def logarithm(PosDict,NegDict,pos,neg,pWords,nWords,extraWords,train):
    totalpos = math.log(pos/(pos+neg))
    totalneg = math.log(neg/(pos+neg))
    for word in train:
        pval = PosDict[word]+1
        nval = NegDict[word]+1
        if(pval==nval==1):
            i = 2
        else:
            totalpos += math.log(pval/(pWords+extraWords))
            totalneg += math.log(nval/(nWords+extraWords))

    if totalpos > totalneg:
        return "pos"
    else:
        return "neg"




if __name__ == "__main__":

    all_docs, all_labels = read_documents("all_sentiment_shuffled.txt")

    split_point = int(0.75*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]  
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    from collections import Counter
    freqsPosWords = Counter()
    freqsNegWords = Counter()
    freqs = Counter()

    import itertools
    poSize = 0
    negSize = 0
    posWords = 0
    negWords = 0
    
    for (doc,label) in zip(train_docs,train_labels):
        if label == "pos":
            for w in doc:
                freqsPosWords[w] += 1
                freqs[w] +=1
                posWords+=1
            poSize +=1
        else:
            for w in doc:
                freqsNegWords[w] += 1
                freqs[w] +=1
                negWords+=1
            negSize +=1
        

    print(len(freqs))

    comp_List = list()


    f= open("Naive-Bayes-dataset.txt","w")
    intCounter = len(train_labels) 
    for truedoc in eval_docs:
        intCounter+=1
        output_string = logarithm(freqsPosWords,freqsNegWords,poSize,negSize,posWords,negWords,len(freqs),truedoc) 
        comp_List.append(output_string)
        f.write("%d,%s\n" % (intCounter,output_string))

    print("pos" in comp_List)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    accuracy = accuracy_score(eval_labels,comp_List)
    f.write("\naccuracy : %s\n" %(accuracy))
    print(accuracy)
    #print(precision_score(eval_labels,comp_List,average="samples"))
    confuse = confusion_matrix(eval_labels,comp_List,labels=["pos","neg"])

    precision = confuse[0][0]/(confuse[0][1] + confuse[0][0])
    recall = confuse[0][0]/(confuse[1][0] + confuse[0][0])
    f1 = 2*precision*recall/(recall+precision)
    
    print(precision)
    print(recall)
    print(f1)
    print(confuse)

    f.write("\nprecision : %s\n" %(str(precision)))
    f.write("\nrecall : %s\n" %(str(recall)))
    f.write("\nf1 : %s\n" %(str(f1)))
    f.write("\nconfusion matrix : \n %s\n" %(confuse))
    f.close()
    


