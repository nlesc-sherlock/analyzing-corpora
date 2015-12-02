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
import matplotlib.cm as cm
from sklearn.manifold import MDS
import gensim
from numpy import argsort

def return_n_words(dic, topic, n_words):
    aa = [(dic[idx],topic[idx]) for idx in argsort(topic)[-n_words:] ]
    return dict(aa)
  
def create_scatter(xs, ys, k_fit, size=100, filename=None):
    num_k = len(set(k_fit))  # number of kernels
    plt.figure(figsize=(15,15))
    x = numpy.arange(num_k)
    yys = [i+x+(i*x)**2 for i in range(num_k)]
    colors = cm.nipy_spectral(numpy.linspace(0, 1, num_k))
    for idx in range(0, num_k):
        plt.scatter(xs[numpy.where(k_fit==idx)], ys[numpy.where(k_fit==idx)], s=100, label=str(idx), c=colors[idx])
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)

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
    create_scatter(pos[:,0], pos[:,1], k_fit, size=100, filename=None)

    dic = gensim.corpora.Dictionary.load("./data/data/filtered_0.1_5_1000000.dic")
    aa = return_n_words(dic, topics[0], 10)
    for idx,cdd in enumerate(aa):
        import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
