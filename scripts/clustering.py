#!/usr/bin/env python
__author__ = 'daniela'
from scipy.spatial.distance import cosine
from corpora.scikit import ScikitLda
import zipfile
import os
import tarfile
import gzip
import zlib
if __name__ == '__main__':

    dirname = "./data/data/enron_out_0.1/"

    for subdir in os.listdir(dirname):
        # print("subdir", subdir)
        for file in os.listdir(dirname+subdir):
            if file.endswith('pkl'):
                lda = ScikitLda.load(dirname+subdir+"/"+file)
                topics = []
                print(lda.topics)
                for topic in lda.topics:
                    topics.append(topic / topic.sum())

                for i in range(len(topics)):
                    for j in range(i + 1, len(topics)):
                        print 'Topic#%d Topic#%d %f' % (i, j, cosine(topics[i], topics[j]))



