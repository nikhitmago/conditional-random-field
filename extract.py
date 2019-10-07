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
        tokens = line.split()
        alnum_tokens = [get_clean_token(token) for token in tokens]
        alnum_tokens = nltk.pos_tag(alnum_tokens)
        final_tokens = map(lambda (x, (y, z)): (x, z), zip(tokens, alnum_tokens))
        data.append(final_tokens)
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

def get_output(a, b):
    start = 0
    end = 0
    output = ''
    for i, val in enumerate(a):
        if i == len(a) - 1:
            sent = ' '.join(b[start:end+1])
            output += '<{}>{}</{}>'.format(val, sent, val)
            break
        if a[i] == a[i+1]:
            end += 1
        else:
            sent = ' '.join(b[start:end+1])
            output += '<{}>{}</{}> '.format(val, sent, val)
            start = i + 1
            end = i + 1
    return output

if __name__ == '__main__': 
    with open(sys.argv[2]) as f:
        test_lines = f.readlines()
    f.close()

    data_test = process_data(test_lines)
    X_test = [sent2features(datum) for datum in data_test]

    tagger = pycrfsuite.Tagger()
    tagger.open(sys.argv[1])
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    with open(sys.argv[3], "w") as f:
        for i in range(len(y_pred)):
            output = get_output(y_pred[i], test_lines[i].split()).strip()
            f.write(output)
            f.write("\n")
    f.close()

