from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from codecs import open
from gensim import corpora 
import pandas as pd
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
            continue
        else:
            totalpos += math.log(pval/(pWords+extraWords))
            totalneg += math.log(nval/(nWords+extraWords))

    if totalpos > totalneg:
        return "pos"
    else:
        return "neg"


def set_scores(eval_labels,comp_List,f):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    accuracy = accuracy_score(eval_labels,comp_List)
    f.write("\naccuracy : %s\n" %(accuracy))
    print(accuracy)

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

if __name__ == "__main__":

    all_docs, all_labels = read_documents("all_sentiment_shuffled.txt")

    choice = input("Enter your tree of choice \n1. Bayes\n2. DT\n3. DT Best\n")

    if choice == "1":

        
        split_point = int(0.80*len(all_docs))
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
            

        comp_List = list()
        f= open("Naive-Bayes-dataset.txt","w")
        intCounter = len(train_labels) 
        for truedoc in eval_docs:
            intCounter+=1
            output_string = logarithm(freqsPosWords,freqsNegWords,poSize,negSize,posWords,negWords,len(freqs),truedoc) 
            comp_List.append(output_string)
            f.write("%d,%s\n" % (intCounter,output_string))

        set_scores(eval_labels,comp_List,f)
        
    elif choice =="2":
        from sklearn import tree
        clf  = tree.DecisionTreeClassifier(criterion="entropy")
        df = pd.DataFrame(all_docs).to_numpy(na_value="")

        # convert your list of lists into token id's. For example, 'aa bb' could be represented as a 2, a as a 1, etc.
        tdict = corpora.Dictionary(df)

        newlist = []
        counter = 0
        for l in df:
            tmplist = []
            for word in l:
                # append to intermediate list the id for the given word under observation
                tmplist.append(tdict.token2id[word])
            # convert to numpy array and append to main list
            newlist.append(np.array(tmplist).astype(float)) # type float
            print(counter)
            counter+=1

        split_point = int(0.80*len(newlist))
        train_docs = newlist[:split_point]
        train_labels = all_labels[:split_point]  
        eval_docs = newlist[split_point:]
        eval_labels = all_labels[split_point:]

        
        clf = clf.fit(train_docs,train_labels)
        comp_List = clf.predict(eval_docs)

        f= open("DT-Entropy-dataset.txt","w")
        intCounter = len(train_labels) 
        for truelabel in comp_List:
            intCounter+=1
            f.write("%d,%s\n" % (intCounter,truelabel))

        set_scores(eval_labels,comp_List,f)
    else:
        from sklearn import tree
        clf  = tree.DecisionTreeClassifier()
        df = pd.DataFrame(all_docs).to_numpy(na_value="")

        # convert your list of lists into token id's. For example, 'aa bb' could be represented as a 2, a as a 1, etc.
        tdict = corpora.Dictionary(df)

        newlist = []
        counter = 0
        for l in df:
            tmplist = []
            for word in l:
                # append to intermediate list the id for the given word under observation
                tmplist.append(tdict.token2id[word])
            # convert to numpy array and append to main list
            newlist.append(np.array(tmplist).astype(float)) # type float
            print(counter)
            counter+=1

        split_point = int(0.80*len(newlist))
        train_docs = newlist[:split_point]
        train_labels = all_labels[:split_point]  
        eval_docs = newlist[split_point:]
        eval_labels = all_labels[split_point:]

        
        clf = clf.fit(train_docs,train_labels)
        comp_List = clf.predict(eval_docs)

        f= open("DT-BEST-dataset.txt","w")
        intCounter = len(train_labels) 
        for truelabel in comp_List:
            intCounter+=1
            f.write("%d,%s\n" % (intCounter,truelabel))

        set_scores(eval_labels,comp_List,f)
    


