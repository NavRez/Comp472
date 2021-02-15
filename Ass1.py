from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
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

def logarithm(PosDict,NegDict,pos,neg,pWords,nWords,extraWords,train,mult):
    totalpos = math.log(pos/(pos+neg))
    totalneg = math.log(neg/(pos+neg))
    for word in train:
        pval = PosDict[word]+1*mult
        nval = NegDict[word]+1*mult
        if(pval==nval==1):
            continue
        else:
            totalpos += math.log(pval/(pWords+int(extraWords*mult)))
            totalneg += math.log(nval/(nWords+int(extraWords*mult)))

    if totalpos > totalneg:
        return "pos"
    else:
        return "neg"

def plotcounter (plotter):
    posC = 0
    negC = 0
    for plot in plotter:
        if plot =="pos":
            posC+=1
        else:
            negC+=1
    return posC,negC

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

    return confuse

def listify(df,tdict):
        newlist = []
        counter = 0
        print("Beginning conversion...")
        for l in df:
            tmplist = []
            for word in l:
                # append to intermediate list the id for the given word under observation
                tmplist.append(tdict.token2id[word])
            # convert to numpy array and append to main list
            newlist.append(np.array(tmplist).astype(float)) # type float
            print("converted line : " + str(counter))
            counter+=1
        return newlist

# taken from https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


if __name__ == "__main__":

    all_docs, all_labels = read_documents("all_sentiment_shuffled.txt")

    
    #setting up matrix and numerical list
    print("setting up matrix ...")
    df = pd.DataFrame(all_docs).to_numpy(na_value="")
    tdict = corpora.Dictionary(df)
    newlist = listify(df,tdict)

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]  
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    ####################### Naive starts
    print("Printing Naive Bayes : acc,precision, recall , f1 and confuse in that order")
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
    
    figes, axes = plt.subplots()
    langs = ['pos','neg']
    students = [poSize,negSize]
    width = 0.35
    ind = np.arange(2)
    rect = axes.bar(np.arange(2), students, width, color='r')

    axes.set_ylabel('Instances')
    axes.set_title('Pos Vs Neg')
    axes.set_xticks(ind)
    axes.set_xticklabels(langs)

    autolabel(rect,axes)

    plt.show(block=False)
    plt.pause(0.001)
    
    comp_List = list()
    f= open("Naive-Bayes-dataset.txt","w")
    intCounter = len(train_labels) 
    for truedoc in eval_docs:
        intCounter+=1
        output_string = logarithm(freqsPosWords,freqsNegWords,poSize,negSize,posWords,negWords,len(freqs),truedoc,1) 
        comp_List.append(output_string)
        f.write("%d,%s\n" % (intCounter,output_string))

    confuseBayes = set_scores(eval_labels,comp_List,f)
    print("Printing of Naive Bayes complete\n\n")

    Bpos,Bneg = plotcounter(comp_List)
    ####################### Naive ends
    
    ### DT delimitation ###

    ####################### DT-Base starts
    print("Printing DT-Base : acc,precision, recall , f1 and confuse in that order")
    clf  = tree.DecisionTreeClassifier(criterion="entropy")

    split_point = int(0.80*len(newlist))
    train_docs = newlist[:split_point]
    eval_docs = newlist[split_point:]

    clf = clf.fit(train_docs,train_labels)
    comp_List = clf.predict(eval_docs)

    f= open("DT-Entropy-dataset.txt","w")
    intCounter = len(train_labels) 
    for truelabel in comp_List:
        intCounter+=1 
        f.write("%d,%s\n" % (intCounter,truelabel))

    confuseDT = set_scores(eval_labels,comp_List,f)
    print("Printing of DT-Base complete\n\n")

    DTpos,DTneg = plotcounter(comp_List)
    ####################### DT-Base ends

    ####################### DT-Best starts
    print("Printing DT-Best : acc,precision, recall , f1 and confuse in that order")

    clf  = tree.DecisionTreeClassifier(criterion="entropy",splitter="random",class_weight="balanced",max_features="auto")
    clf = clf.fit(train_docs,train_labels)
    comp_List = clf.predict(eval_docs)

    f= open("DT-BEST-dataset.txt","w")
    intCounter = len(train_labels) 
    for truelabel in comp_List:
        intCounter+=1
        f.write("%d,%s\n" % (intCounter,truelabel))

    confuseDTB = set_scores(eval_labels,comp_List,f)
    print("Printing DT-Best complete")

    DTBestpos,DTBestneg = plotcounter(comp_List)

    evalPos,evalNeg = plotcounter(eval_labels)

    objects =("Bayes", "DT", "DTBest", "Real")
    width = 0.35 
    ind = np.arange(4)
    performPOs = (Bpos,DTpos,DTBestpos,evalPos)
    performNEg = (Bneg,DTneg,DTBestneg,evalNeg)
    errorPerfPos = (confuseBayes[0][0],confuseDT[0][0],confuseDTB[0][0],0)
    errorPerfNeg = (confuseBayes[1][1],confuseDT[1][1],confuseDTB[1][1],0)

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, performPOs, width, color='g')
    rects2 = ax.bar(ind + width, performNEg, width, color='r')
    rects3 = ax.bar(ind, errorPerfPos, width, color='b')
    rects4 = ax.bar(ind + width, errorPerfNeg, width, color='y')

    ax.set_ylabel('Instances')
    ax.set_title('Instance by Model')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(objects)

    ax.legend((rects1[0], rects2[0],rects3[0], rects4[0]), ('Pos', 'Neg',"True Pos","True Neg"))

    plt.grid(True)
    plt.show(block=False)
    plt.pause(10)
    ####################### DT-Base ends
    


