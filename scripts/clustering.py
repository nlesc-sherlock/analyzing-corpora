#!/usr/bin/env python
__author__ = 'daniela'
from scipy.spatial.distance import cosine
from corpora.scikit import ScikitLda
import zipfile
import os
import tarfile
if __name__ == '__main__':

    dirname = "/home/daniela/git/analyzing-corpora/data/data/enron_out_0.1/"

    for filename in os.listdir(dirname):
        # archive = zipfile.ZipFile(dirname+filename, mode='r')
        tar = tarfile.open(dirname+filename, "r:bz2")
        # print(tar.name)
        for tarinfo in tar:
            print tarinfo.name, "is", tarinfo.size, "bytes in size and is",
            if tarinfo.isreg():
                f = tar.extractfile(tarinfo)
                print(f.name)
                lda = ScikitLda.load(dirname+f)
                # lda = ScikitLda.load(f)


    # lda = ScikitLda.load("/home/daniela/git/analyzing-corpora/data/data/enron_out_0.1/lda_4/lda_4.pkl")
    #
    # topics = []
    # for topic in lda.topics:
    #     topics.append(topic / topic.sum())
    #
    # for i in range(len(topics)):
    #     for j in range(i + 1, len(topics)):
    #         print 'Topic#%d Topic#%d %f' % (i, j, cosine(topics[i], topics[j]))



