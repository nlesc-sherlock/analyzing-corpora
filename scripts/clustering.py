#!/usr/bin/env python
__author__ = 'daniela'
from scipy.spatial.distance import cosine
from corpora.scikit import ScikitLda

import os
import zlib
import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import gensim
from numpy import argsort

def return_n_words(dic, topic, n_words):
  aa = [(dic[idx],topic[idx]) for idx in argsort(topic)[-n_words:] ]
  return dict(aa)
  
if __name__ == '__main__':
    dirname = "./data/data/enron_out_0.1/"
    topics = []
    for subdir in [x[0] for x in os.walk(dirname)][1:3]:

        for file in os.listdir(subdir):
            if file.endswith('pkl'):
                print("attempting... ", file)
                lda = ScikitLda.load(subdir+"/"+file)
                for topic in lda.topics:
                    topics.append(topic / topic.sum())
                    
    #for i in range(len(topics)):
    #  for j in range(i + 1, len(topics)):
                #            print 'Topic#%d Topic#%d %f' % (i, j, cosine(topics[i], topics[j]))

    cos_distance = pairwise_distances(topics, metric='cosine')
    k_fit = KMeans(n_clusters=13).fit_predict(cos_distance)
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(cos_distance).embedding_
    plt.scatter(pos[:,0],pos[:,1], c=k_fit, s=100)
    plt.show()
    dic = gensim.corpora.Dictionary.load("./data/data/filtered_0.1_5_1000000.dic")
    aa = return_n_words(dic, topics[0], 10)

