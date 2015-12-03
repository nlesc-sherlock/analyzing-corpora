#!/usr/bin/env python

# Copyright 2015 Netherlands eScience Center <info@esciencecenter.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from sklearn.manifold import MDS, TSNE
import gensim
from numpy import argsort
from wordcloud import WordCloud

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
        plt.scatter(xs[numpy.where(k_fit==idx)], ys[numpy.where(k_fit==idx)],
                    s=100, label=str(idx), c=colors[idx])
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
    plt.close()

def create_wordcloud(topic_weights, filename=None):
    plt.figure(figsize=(16,40))
    for idx,topic in enumerate(topic_weights):
        wc = WordCloud(background_color="white")
        ww = [(word,weight) for word,weight in topic.iteritems()]
        img = wc.generate_from_frequencies(ww)
        plt.subplot(len(topic_weights),2,2*idx+1)
        plt.axis('off')
        plt.imshow(img)
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
    plt.close()

def find_topics_per_cluster(topics, k_fit, cluster):
    num_k = len(set(k_fit))  # number of kernels
    cluster_indices = [ numpy.where(k_fit==n) for n in range(0, num_k) ]
    topic_out = [topics[n] for n in cluster_indices[cluster][0]]
    return topic_out

def load_topics(dirname):
    topics = []
    for subdir in [x[0] for x in os.walk(dirname)][1:]:
        for file in os.listdir(subdir):
            if file.endswith('pkl'):
                print("attempting... ", file)
                lda = ScikitLda.load(subdir+"/"+file)
                for topic in lda.topics:
                    topics.append(topic / topic.sum())
    return topics

def find_distance_matrix(topics, metric='cosine'):
    if metric == 'cosine':
        distance_matrix = pairwise_distances(topics, metric='cosine')
        # diagonals should be exactly zero
        numpy.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def data_embedding(distance_matrix, type='TSNE'):
    if type == 'TSNE':
        model = TSNE(n_components=2, metric='precomputed')
    if type == 'MDS':
        model = MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    pos = model.fit(distance_matrix).embedding_  # position of points in embedding space
    return pos

def Wk(mu, clusters):
    K = len(mu)
    return sum([numpy.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

# finding optimum k in k means cluster based on
# https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], numpy.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(numpy.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = numpy.array([numpy.random.normal(c[0], s), numpy.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = numpy.array(X)[:N]
    return X

if __name__ == '__main__':
    dirname = "./data/data/enron_out_0.1/"
    topics = load_topics(dirname)
    distance_matrix = find_distance_matrix(topics, metric='cosine')
    k_fit = KMeans(n_clusters=25).fit_predict(distance_matrix)
    pos = data_embedding(distance_matrix, type='TSNE')
    create_scatter(pos[:,0], pos[:,1], k_fit, size=100, filename=None)
    dic = gensim.corpora.Dictionary.load("./data/data/filtered_0.1_5_1000000.dic")
    topic_out = find_topics_per_cluster(topics, k_fit, 2)
    topic_weights = [ return_n_words(dic, topic_out[idx], 25) for idx in range(0,len(topic_out[:])) ]
    create_wordcloud(topic_weights, filename=None)

