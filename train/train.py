import sys
import numpy as np
import nltk
import re
import pycrfsuite
from bs4 import BeautifulSoup
import json
from itertools import chain
from operator import itemgetter 

def get_clean_token(token):
    if token.isalnum():
        return token
    else:
        if (not token[0].isalnum()) and (not token[-1].isalnum()):
            return token[1:-1]
        elif not token[0].isalnum():
            return token[1:]
        elif not token[-1].isalnum():
            return token[:-1]
        else:
            return token

def process_data(lines):
    data = []
    for line in lines:
        line = line.strip()
        soup = BeautifulSoup(line, "html.parser")
        datum = []
        for label in [tag.name for tag in soup.find_all()]:
            for elem in soup.find_all(label):
                tokens = elem.text.split()
                alnum_tokens = [get_clean_token(token) for token in tokens]
                alnum_tokens = nltk.pos_tag(alnum_tokens)
                final_tokens = map(lambda (x, (y, z)): (x, z), zip(tokens, alnum_tokens))
                final_tokens = map(lambda (x, y): (x, y, label), final_tokens)
                datum += final_tokens
        data.append(datum)
    return data

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
    ]
    if i == 0:
        features.append('BOS')
    if i == len(sent)-1:
        features.append('EOS')
        
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
        ])
    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.extend([
            '-2:word.lower=' + word2.lower(),
            '-2:word.isupper=%s' % word2.isupper(),
            '-2:postag=' + postag2,
        ])
    if i < len(sent)-1:
        word_1 = sent[i+1][0]
        postag_1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word_1.lower(),
            '+1:word.isupper=%s' % word_1.isupper(),
            '+1:postag=' + postag_1,
        ])
    if i < len(sent)-2:
        word_2 = sent[i+2][0]
        postag_2 = sent[i+2][1]
        features.extend([
            '+2:word.lower=' + word_2.lower(),
            '+2:word.isupper=%s' % word_2.isupper(),
            '+2:postag=' + postag_2,
        ]) 
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def get_confusion_matrix(a, b, tag, tp, fp, fn):
    d = []
    chunk = []
    for i, val in enumerate(a):
        if val == tag:
            chunk.append(i)
        else:
            if chunk:
                d.append(chunk)
                chunk = []
    if chunk:
        d.append(chunk)

    for block in d:
        if itemgetter(*block)(b) == itemgetter(*block)(a):
            tp += 1
        else:
            if len(np.unique(itemgetter(*block)(b))) == 1:
                fn += 1
            else:
                fp += 1
    
    return tp, fp, fn

def get_scores(y_test, y_pred, is_test):
    if is_test:
        print("Test Data Performance Report\n")
    else:
        print("Train Data Performance Report\n")
    tags = ['format', 'requisite', 'description', 'grading', 'others']
    for tag in tags:
        tp = 0
        fp = 0
        fn = 0
        for i, pred in enumerate(y_pred):
            tp, fp, fn = get_confusion_matrix(y_test[i], pred, tag, tp, fp, fn)
        recall = np.round(float(tp)/(tp+fn), 2)
        prec = np.round(float(tp)/(tp+fp), 2)
        f1 = np.round((2.0*recall*prec)/(recall + prec), 2)
        print("Tag -> {}".format(tag))
        print("F1 score = {}".format(f1))
        print("Recall score = {}".format(recall))
        print("Precision score = {}".format(prec))
        print("\n")

if __name__ == '__main__':  
    with open('train-ucla.txt') as f:
        train_lines = f.readlines()
    f.close()

    with open('ground-truth-test.txt') as f:
        test_lines = f.readlines()
    f.close()

    data_train = process_data(train_lines)
    X_train = [sent2features(datum) for datum in data_train]
    y_train = [sent2labels(datum) for datum in data_train]

    data_test = process_data(test_lines)
    X_test = [sent2features(datum) for datum in data_test]
    y_test = [sent2labels(datum) for datum in data_test]

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train('ucla.model')

    tagger = pycrfsuite.Tagger()
    tagger.open('ucla.model')

    #Train set report
    y_pred = [tagger.tag(xseq) for xseq in X_train]
    get_scores(y_train, y_pred, False)

    #Test set report
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    get_scores(y_test, y_pred, True)

