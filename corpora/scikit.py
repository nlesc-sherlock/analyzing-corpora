from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from matplotlib.pyplot import savefig, subplot, figure, imshow, plot, axis, title
from corpus import load_vraagtekst_corpus, Corpus

class ScikitLda(object):
    def __init__(self, corpus, n_topics, max_iter=5, learning_method='online',
                 learning_offset=50., **kwargs):
        self.lda = LatentDirichletAllocation(n_topics=n_topics,
                                             max_iter=max_iter,
                                             learning_method=learning_method,
                                             learning_offset=learning_offset,
                                             **kwargs)
        self._corpus = corpus
        self.lda.fit(corpus.sparse_matrix())
        self._weights = None

    @property
    def topics(self):
        return self.lda.components_

    @property
    def n_topics(self):
        return self.lda.n_topics

    @property
    def corpus(self):
        return self._corpus

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self.lda.transform(self.corpus.sparse_matrix())
            self._weights = (self._weights.T / self._weights.sum(axis=1)).T
        return self._weights

    def transform(self, corpus):
        return self.lda.transform(corpus.sparse_matrix())

    def topic_words(self, n_words=10):
        topicWords = []
        topicWeightedWords = []

        for topic_idx, topic in enumerate(self.topics):
            weightedWordIdx = topic.argsort()[::-1]
            wordsInTopic = [self.corpus.word(i) for i in weightedWordIdx[:n_words]]

            weights = topic / topic.sum()
            topicWeights = [(weights[i], self.corpus.word(i))
                            for i in weightedWordIdx[:n_words]]

            topicWords.append(wordsInTopic)
            topicWeightedWords.append(topicWeights)

        return (topicWords, topicWeightedWords)


def topics_by_discrete_property(lda, all_property_values):
    values = np.unique(all_property_values)
    topicsByProperty = np.empty((len(values), lda.n_topics))

    for i, prop_value in enumerate(values):
        mask = np.asarray(prop_value == all_property_values)
        prop_weights = lda.weights[mask]
        topicsByProperty[i] = np.average(prop_weights, axis=0)

    return topicsByProperty, values


def topics_by_integer_property(lda, all_property_values, delta = 5):
    all_property_values = np.array(all_property_values)
    size = int(np.nanmax(all_property_values) + 1)
    topicsByProperty = np.zeros((size, lda.n_topics))

    lower = all_property_values - delta
    upper = all_property_values + delta
    for prop_value in np.arange(size):
        mask = (prop_value >= lower) & (prop_value <= upper)
        prop_weights = lda.weights[mask]
        if len(prop_weights) > 0:
            topicsByProperty[prop_value] = np.average(prop_weights, axis=0)

    return topicsByProperty


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

def select_top(array, n):
    return array.argsort()[:-n-1:-1], np.sort(array)[:-n-1:-1]

if __name__ == '__main__':
    corpus = load_vraagtekst_corpus('data/preprocessedData.pkl')

    print("nSamples (docs) : {0}".format(corpus.num_samples))
    print("nFeatures(words): {0}".format(corpus.num_features))

    lda = ScikitLda(corpus, n_topics=10)
    topicWords, topicWeightedWords = lda.topic_words()

    for topic_idx, wordsInTopic in enumerate(topicWords):
        print "Topic #%d:" % topic_idx
        print " ".join(wordsInTopic)

    topicsByOrg, orgs = topics_by_discrete_property(lda, corpus.metadata_frame['individu of groep'])
    averageWeights = np.average(lda.weights, axis=0)
    # get topic specificity by comparing with the average topic weights
    # normalize by average topic weights
    topicsSpecificityByOrg = (topicsByOrg - averageWeights) / averageWeights

    for i, org in enumerate(orgs):
        print("Organisation {0}:".format(org))
        prev, ratio = select_top(topicsByOrg[i], 3)
        print("\tPrevalent topics: {0} (ratio: {1})".format(prev, ratio))
        prev, ratio = select_top(topicsSpecificityByOrg[i], 3)
        print("\tSpecific topics: {0} (ratio: {1})".format(prev, ratio))

    topicsByAge = topics_by_integer_property(lda, corpus.metadata_frame['Leeftijd'])
    plot_wordcloud_with_property(topicWeightedWords, topicsByAge)
    savefig('topics.png')
