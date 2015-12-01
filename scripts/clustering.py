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
    for subdir in [x[0] for x in os.walk(dirname)][1:]:
        # print("subdir", subdir)
        for file in os.listdir(subdir):
            if file.endswith('pkl'):
                print("attempting... ", file)

                lda = ScikitLda.load(subdir+"/"+file)
                topics = []
                print "TOPICS in file %s " %file
                print lda.topics

                for topic in lda.topics:
                    topics.append(topic / topic.sum())

                for i in range(len(topics)):
                    for j in range(i + 1, len(topics)):
                            print 'Topic#%d Topic#%d %f' % (i, j, cosine(topics[i], topics[j]))



