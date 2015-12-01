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
from scipy import sparse
import time

if __name__ == '__main__':
    #t = timeit.Timer("print 'main statement'", "print 'setup'")
    dirname = "./data/data/enron_out_0.1/"
    topics = []
    for subdir in [x[0] for x in os.walk(dirname)][1:]:
        # print("subdir", subdir)
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
    import pdb; pdb.set_trace()
    
#KMeans(n_clusters=2).fit(lda.topics)
