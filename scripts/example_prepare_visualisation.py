#!/usr/bin/python
from prepare_for_visualisation import prepare_for_visualisation


utomat = 'data/enron_lda_15_norm.csv'
dictionary = 'data/enron_dic.csv'
delim = ','
allWords, utoorder, utonorm, winninguto, Nwordsto = prepare_for_visualisation(utomat,dictionary,delim)
print allWords.shape
