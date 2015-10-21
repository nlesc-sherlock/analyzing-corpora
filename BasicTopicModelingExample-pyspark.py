# Run as:
#  $ $SPARK_HOME/bin/pyspark BasicTopicModelingExample-pyspark.py
from pyspark import SparkContext

from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

from scipy.sparse import csr_matrix

from pyspark.mllib.linalg import Matrix
from pyspark.mllib.linalg import Matrices

import pandas as pd
from wordcloud import WordCloud
import gensim

import numpy as np
import matplotlib.pyplot as plt

sc = SparkContext("local", "Simple App1")

# # Load data_vraag from file

import pickle
data_vraag = pickle.load(open('preprocessedData.pkl', 'r'))


data_ppl = data_vraag[data_vraag['individu of groep']=='mijzelf']
data_org = data_vraag[data_vraag['individu of groep']!='mijzelf']

vraagTokens = data_vraag['SentToks'].tolist()

dic = gensim.corpora.Dictionary(vraagTokens)
corpus = [dic.doc2bow(text) for text in vraagTokens]

def lineToSparse(line):
    v = [float(x) for x in line.strip().split(' ')]
    v = { idx: v[idx] for idx,val in zip(np.nonzero(v)[0], v) }
    v = Vectors.sparse(11, v)
    return v

def toSparseVector(corpusLine, nFeatures):
    v = { idx: val for idx,val in corpusLine }
    return Vectors.sparse(nFeatures, v)

# nSamples = len(corpus)
nFeatures = len(dic)

corpusParallel = sc.parallelize(corpus)
corpusMapped = corpusParallel.map(lambda doc: toSparseVector(doc, nFeatures))

# Index documents with unique IDs
corpusIndexed = corpusMapped.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

nTopics = 10
ldaModel = LDA.train(corpusIndexed, k=nTopics)

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=nTopics, max_iter=1,
                                learning_method='online', learning_offset=50.
                               )
doc0 = corpusIndexed.first()[1].toArray()
lda.fit(doc0)
lda.components_ = ldaModel.topicsMatrix().T


def getDocumentTopics(docTokens, lda):
    wcTuples = dic.doc2bow(docTokens)
    data = []
    row  = []
    col  = []

    for w,c in wcTuples:
        col.append(0)
        row.append(w)
        data.append(c)

    nSamples = 1
    nFeatures = len(dic)
    oneDoc = csr_matrix((data, (col,row)), shape=(nSamples, nFeatures))
    docWeights = lda.transform(oneDoc)[0]
    docWeights /= docWeights.sum()
    return docWeights


def inRange(age, targetAge, delta):
    return (targetAge-delta)<=age and age<=(targetAge+delta)

def getPplCirca(targetAge, delta):
    return data_ppl[data_ppl['Leeftijd'].apply(lambda age: inRange(age,targetAge, delta))]


topicsByAge = np.zeros((data_ppl['Leeftijd'].max()+1, nTopics))
deltaAge = 5

for age in np.arange(data_ppl['Leeftijd'].max()+1):
    dataGroup = getPplCirca(age,deltaAge)
    groupTokens = dataGroup['SentToks'].tolist()

    for qTokens in groupTokens:
        topicWeights = getDocumentTopics(qTokens, lda)
        for topic,weight in enumerate(topicWeights):
            topicsByAge[age,topic] += weight / len(groupTokens)

topicWords = []
topicWeightedWords = []

for topic_idx, topic in enumerate(lda.components_):
    weightedWordIdx = topic.argsort()[::-1]
    wordsInTopic = [dic[i] for i in weightedWordIdx[:10]]

    weights = topic / topic.sum()
    topicWeights = [ (weights[i],dic[i]) for i in weightedWordIdx[:10]]

    print "Topic #%d:" % topic_idx
    print " ".join(wordsInTopic)
    topicWords.append(wordsInTopic)
    topicWeightedWords.append(topicWeights)

plt.figure(figsize=(16,40))
for idx,topic in enumerate(topicWeightedWords):
    wc = WordCloud(background_color="white")
    img = wc.generate_from_frequencies([ (word, weight) for weight,word in topic ])
    plt.subplot(nTopics,2,2*idx+1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(nTopics,2,2*idx+2)
    plt.plot(topicsByAge[:,idx])
    plt.axis([10, 100, 0, 1.0])
    plt.title('Topic #%2d'%(idx))
plt.show()
