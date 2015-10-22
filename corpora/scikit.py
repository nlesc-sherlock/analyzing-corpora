from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from matplotlib.pyplot import savefig, subplot, figure, imshow, plot, axis, title
from corpus import load_vraagtekst_corpus, Corpus

def scikit_lda(mm, n_topics):
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.
                                #,random_state=0
                               )
    lda.fit(mm)
    return lda

def topic_words(lda, corpus):
    topicWords = []
    topicWeightedWords = []

    for topic_idx, topic in enumerate(lda.components_):
        weightedWordIdx = topic.argsort()[::-1]
        wordsInTopic = [corpus.word(i) for i in weightedWordIdx[:10]]

        weights = topic / topic.sum()
        topicWeights = [(weights[i], corpus.word(i)) for i in weightedWordIdx[:10]]
        
        print "Topic #%d:" % topic_idx
        print " ".join(wordsInTopic)
        topicWords.append(wordsInTopic)
        topicWeightedWords.append(topicWeights)

    return (topicWords, topicWeightedWords)


def topics_by_numeric_property(lda, corpus, metadata_property, delta = 5):
    topicsByProperty = np.zeros((metadata_property.max()+1, lda.n_topics))

    for prop_value in np.arange(metadata_property.max()+1): 
        indexes = np.where((metadata_property >= prop_value - delta) &
                           (metadata_property <= prop_value + delta))[0]
        
        for idx in indexes:
            topicWeights = getDocumentTopics(lda, corpus.with_index(idx))
            for topic, weight in enumerate(topicWeights):
                topicsByProperty[prop_value, topic] += weight / len(indexes)

    return topicsByProperty


def topics_by_discrete_property(lda, corpus, metadata_property):
    values = np.unique(metadata_property)
    topicsByProperty = np.empty((len(values), lda.n_topics))
    allWeights = np.asarray(lda.transform(corpus.sparse_matrix()))
    allWeights = (allWeights.T / allWeights.sum(axis=1)).T

    for i, prop_value in enumerate(values):
        prop_weights = allWeights[metadata_property == prop_value]
        topicsByProperty[i] = np.average(prop_weights, axis=0)

    return (topicsByProperty, values)

def plot_wordcloud_with_property(topicWeightedWords, topicsByProperty):
    figure(figsize=(16,40))
    for idx,topic in enumerate(topicWeightedWords):
        wc = WordCloud(background_color="white")
        img = wc.generate_from_frequencies([ (word, weight) for weight,word in topic ])
        subplot(len(topicWeightedWords),2,2*idx+1)
        imshow(img)
        axis('off')
        
        subplot(len(topicWeightedWords),2,2*idx+2)
        plot(topicsByProperty[:,idx])
        axis([10, 100, 0, 1.0])
        title('Topic #%2d'%(idx))

def getDocumentTopics(lda, corpus):
    docWeights = lda.transform(corpus.sparse_matrix())[0]
    return docWeights / docWeights.sum()

if __name__ == '__main__':
    corpus = load_vraagtekst_corpus('preprocessedData.pkl')

    print("nSamples (docs) : {0}".format(corpus.num_samples))
    print("nFeatures(words): {0}".format(corpus.num_features))

    lda = scikit_lda(corpus.sparse_matrix(), n_topics=10)
    topicWords, topicWeightedWords = topic_words(lda, corpus)
    topicsByOrg, orgs = topics_by_discrete_property(lda, corpus, corpus.metadata_frame['individu of groep'])
    for i, org in enumerate(orgs):
        print org, '-', topicWords[np.argmax(topicsByOrg[i])][0]
    topicsByAge = topics_by_numeric_property(lda, corpus, corpus.metadata_frame['Leeftijd'])
    plot_wordcloud_with_property(topicWeightedWords, topicsByAge)
    savefig('topics.png')
