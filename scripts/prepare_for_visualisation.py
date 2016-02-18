import numpy as np
from collections import Counter

def prepare_for_visualisation(utomat,dictionary,delim):
    """
    input:
    - unittopicmat:  path to csv file with unit (e.g. a word or a document) by topic matrix
    - dictionary:  path to csv file with dictionrary
    - delim: delimiter used in csv file
    output:
    - allWords: N units by N topics array with ranking of units
    - utoorder: N units by N topics array with ranking of unit indices
    - utonorm: N units by N topics array with normalized probabilities
    - winninguto: N units array with winning topic per unit
    - Nwordsto: N topic array with number of winning units per topic
    """

    # load data
    with open(dictionary, 'r') as fin: #'data/enron_dic.csv'
        lines = fin.readlines()
    tmpLines = [ line.strip().split('\t') for line in lines ]
    dic = { l[0]: l[1] for l in tmpLines }
    uto = np.loadtxt(utomat, delimiter=delim) #','
    RS1 = uto.sum(axis=1)
    #normalized word by topic array
    utonorm = uto / RS1[:,None]
    #winning topic per word
    winninguto = utonorm.argmax(axis=1)
    #frequency table for number of words that are most popular per topic

    Nwordsto = Counter(winninguto)
    # now calculate order of word within each topic (output index)
    utoorder = np.zeros(shape=utonorm.shape)
    utoorder_words = np.zeros(shape=utonorm.shape)

    idx = utonorm.argsort(axis=0)
    utoorder = idx[::-1]
    # now calculate order of words within each topic (output words)
    allWords = []
    for j in range(utoorder.shape[1]):
        topic = utoorder[:,j]
        tmp = [ dic[str(wordIdx)] for wordIdx in topic ]
        allWords.append(tmp)
    allWords = np.array(allWords).T
    return allWords, utoorder, utonorm, winninguto, Nwordsto
