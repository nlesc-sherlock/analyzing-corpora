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

from corpora.scikit import ScikitLda

import os
import numpy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS, TSNE
import gensim
from numpy import argsort
from wordcloud import WordCloud
import math
import argparse


class clustering:

    def __init__(self, dirname, dictionary):
        # initialize object
        self.dirname = dirname
        self.dictionary = dictionary
        self.load_topics(self.dirname)
        self.dic = gensim.corpora.Dictionary.load(self.dictionary)
        self.find_distance_matrix(metric='cosine')
        self.angularize()  # angularize distance matrix
        self.pos = self.data_embedding(type='TSNE')

        # self.main()

    def return_n_words(self, dic, topic, n_words):
        '''
        return the top n words in the topic
        '''
        aa = [(dic[idx], topic[idx]) for idx in argsort(topic)[-n_words:]]
        return dict(aa)

    def create_scatter(self, size=100, filename=None):
        '''
        create scatter plot of the clusters found
        '''
        num_k = len(set(self.k_fit))  # number of kernels
        plt.figure(figsize=(15, 15))
        x = numpy.arange(num_k)
        # TODO: yys is unused!!
        yys = [i + x + (i * x)**2 for i in range(num_k)]
        colors = cm.nipy_spectral(numpy.linspace(0, 1, num_k))
        for idx in range(0, num_k):
            plt.scatter(self.pos[numpy.where(self.k_fit == idx), 0], self.pos[numpy.where(self.k_fit == idx), 1],
                        s=100, label=str(idx), c=colors[idx])
        plt.legend()
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300)
        plt.close()

    def create_wordcloud(self, filename=None):
        '''
        create a wordcloud of the top words in a cluster
        '''
        plt.figure()
        for idx, topic in enumerate(self.topic_weights):
            wc = WordCloud(background_color="white")
            ww = [(word, weight) for word, weight in topic.iteritems()]
            img = wc.generate_from_frequencies(ww)
            plt.subplot(len(self.topic_weights), 2, 2 * idx + 1)
            plt.axis('off')
            plt.imshow(img)
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300)
        plt.close()

    def find_topics_per_cluster(self, topics, k_fit, cluster):
        '''
        return the topics in a given cluster
        '''
        num_k = len(set(k_fit))  # number of kernels
        cluster_indices = [numpy.where(k_fit == n) for n in range(0, num_k)]
        topic_out = [topics[n] for n in cluster_indices[cluster][0]]
        return topic_out

    def load_topics(self, dirname):
        self.topics = []
        for subdir in [x[0] for x in os.walk(dirname)][1:]:
            for file in os.listdir(subdir):
                if file.endswith('pkl'):
                    print("attempting... ", file)
                    lda = ScikitLda.load(subdir + "/" + file)
                    for topic in lda.topics:
                        self.topics.append(topic / topic.sum())

    def find_distance_matrix(self, metric='cosine'):
        '''
        compute distance matrix between topis using cosine or euclidean
        distance (default=cosine distance)
        '''
        if metric == 'cosine':
            self.distance_matrix = pairwise_distances(self.topics,
                                                      metric='cosine')
            # diagonals should be exactly zero, so remove rounding errors
            numpy.fill_diagonal(self.distance_matrix, 0)
        if metric == 'euclidean':
            self.distance_matrix = pairwise_distances(self.topics,
                                                      metric='euclidean')

    def data_embedding(self, type='TSNE'):
        '''
        Fit distance matrix into two-dimensions embedded space using
        the TSNE or MDS model
        '''
        if type == 'TSNE':
            model = TSNE(n_components=2, metric='precomputed')
        if type == 'MDS':
            model = MDS(n_components=2, max_iter=3000, eps=1e-9,
                        dissimilarity="precomputed", n_jobs=1)
        # position of points in embedding space
        pos = model.fit(self.distance_matrix).embedding_
        return pos

    def explained_variance(self, nclusters, filename='elbow.pdf'):
        '''
        calculate explained variance for a range of number of clusers
        defined by 1:nclusters
        plot elbow curve of explained variance against number of clusters
        '''
        from scipy.spatial.distance import cdist, pdist
        KK = range(1, nclusters + 1)
        centers = [KMeans(n_clusters=k).fit(
            self.distance_matrix).cluster_centers_ for k in KK]
        D_k = [cdist(self.distance_matrix, cent, 'euclidean')
               for cent in centers]
        dist = [numpy.min(D, axis=1) for D in D_k]
        # Total within-cluster sum of squares
        tot_withinss = [sum(d**2) for d in dist]
        # The total sum of squares
        totss = sum(pdist(self.distance_matrix)**2) / \
            self.distance_matrix.shape[0]
        # The between-cluster sum of squares
        betweenss = totss - tot_withinss
        # elbow curve
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(KK, betweenss / totss * 100, 'b*-')
        ax.set_ylim((0, 100))
        plt.grid(True)
        plt.xlabel('Number of clusters')
        plt.ylabel('Percentage of variance explained (%)')
        plt.title('Elbow for KMeans clustering')
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300)
        plt.close()

    def computeAngle(self, x):
        '''
        description
        '''
        result = math.fabs(2 * math.acos(1 - math.fabs(x)) / math.pi)
        return result

    def angularize(self):
        '''
        angularize distance matrix
        '''
        for i in range(len(self.distance_matrix)):
            for j in range(len(self.distance_matrix[0])):
                v = self.computeAngle(self.distance_matrix[i][j])
                self.distance_matrix[i][j] = v

    def kmeans(self, n_clusters):
        '''
        run KMeans clustering algorithm with n_clusters number of clusters
        '''
        self.k_fit = KMeans(n_clusters=20,
                            precompute_distances=True).fit_predict(
            self.distance_matrix)
        self.remap()

    def dbscan(self, min_samples=3, eps=0.56):
        '''
        run DBSCAN clustering algorithm
        '''
        self.k_fit = DBSCAN(min_samples=min_samples, eps=eps,
                            metric='precomputed').fit_predict(
            self.distance_matrix)
        self.remap()

    def remap(self):
        '''
        remap cluster -1 to max cluster + 1
        '''
        self.k_fit[self.k_fit == -1] = max(self.k_fit) + 1

    def find_largest_cluster(self):
        '''
        return the largest cluster
        '''
        return max(set(self.k_fit), key=list(self.k_fit).count)

    def topic_weights(self, cluster, nwords=25):
        '''
        find topic_weights for a given cluster
        '''
        # find largest cluster
        topic_out = self.find_topics_per_cluster(
            self.topics, self.k_fit, cluster)
        self.topic_weights = [self.return_n_words(self.dic, topic_out[idx], 25)
                              for idx in range(0, len(topic_out[:]))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic model clustering')
    parser.add_argument('--dirname', help='Directory with...')
    parser.add_argument('--dictionary', help='gensim dictionary')
    parser.add_argument('--cluster', type=str, help='clustering type')
    parser.add_argument('--matrix', type=str, default='cosine',
                        help='distance matrix type [default=cosine]')
    args = parser.parse_args()
    # initialize object
    cluster = clustering(args.dirname, args.dictionary)
    # calculated explained variance for a range of clusters 1:nclusters
    # (plots elbow curve)
    cluster.explained_variance(nclusters=10, filename='elbow.pdf')
    # cluster.kmeans(10)
    cluster.dbscan(min_samples=3, eps=0.56)
    import pdb
    pdb.set_trace()
    # find largest cluster
    largest_cluster = cluster.find_largest_cluster()
    # create output plots
    cluster.topic_weights(largest_cluster)
    cluster.create_wordcloud()
    cluster.create_scatter()
