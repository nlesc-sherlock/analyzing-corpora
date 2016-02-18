import numpy as np
from collections import Counter

def prepare_for_visualisation(utomat,dictionary,delim=',',maxword=10):
    """
    input:
    - unittopicmat:  path to csv file with unit (e.g. a word or a document) by topic matrix
    - dictionary:  path to csv file with dictionrary
    - delim: delimiter used in csv file
    - maxword: maximum number of words to output in the dictionary
    output:
    - wpdict: dictionary with most popular words and their probability per topic
    # turned off: - allWords: N units by N topics array with ranking of units
    # turned off: - utoorder: N units by N topics array with ranking of unit indices
    # turned off: - utonorm: N units by N topics array with normalized probabilities
    # turned off: - winninguto: N units array with winning topic per unit
    # turned off: - Nwordsto: N topic array with number of winning units per topic
    """
    # load data
    with open(dictionary, 'r') as fin: #'data/enron_dic.csv'
        lines = fin.readlines()
    tmpLines = [ line.strip().split('\t') for line in lines ]
    dic = { l[0]: l[1] for l in tmpLines }
    uto = np.loadtxt(utomat, delimiter=delim) #','
    RS1 = uto.sum(axis=1)
    RS2 = uto.sum(axis=0)
    #normalized word by topic array
    utonorm = uto / RS1[:,None]
    utonorm = utonorm / RS2[None,:]
    #winning topic per word
    winninguto = utonorm.argmax(axis=1)
    #frequency table for number of words that are most popular per topic
    Nwordsto = Counter(winninguto)
    # now calculate order of word within each topic (output index)
    utoorder = np.zeros(shape=utonorm.shape)
    utoorder_words = np.zeros(shape=utonorm.shape)
    idx = utonorm.argsort(axis=0)
    utoorder = idx[::-1]
    utonorm = np.around(utonorm,decimals=3)
    # now calculate order of words within each topic (output words)
    allWords = []
    for j in range(utoorder.shape[1]):
        topic = utoorder[:,j]
        tmp = [ dic[str(wordIdx)] for wordIdx in topic ]
        allWords.append(tmp)
        #allWords = np.array(allWords).T
        allProbs = []
    for j in range(utoorder.shape[1]):
        topic = utoorder[:,j]
        tmp = [ utonorm[wordIdx,j] for wordIdx in topic ]
        tmp = np.around(1- np.cumsum(tmp),decimals=3)
        allProbs.append(tmp)
    #merge probabilities and words into one large tuple
    allProbs_flat = [y for x in allProbs for y in x]
    allWords_flat = [y for x in allWords for y in x]
    output = zip(allProbs_flat,allWords_flat)
    # built dictionary of topics, words and probabilities
    wpdict = {}
    ntopics = utonorm.shape[1]
    nw = utonorm.shape[0]
    if maxword > nw:
        maxword = nw
    for i in range(ntopics):
        i0 = i*nw
        wpdict["Topic %i" %(i+1) ] = output[i0:(i0+maxword-1)]
        # return allWords, utoorder, utonorm, winninguto, Nwordsto, wpdict
    return wpdict
