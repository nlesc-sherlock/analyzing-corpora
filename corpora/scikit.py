#!/usr/bin/env python
# SIM-CITY client
#
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

from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from matplotlib.pyplot import subplot, figure, imshow, plot, axis, title


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
            wordsInTopic = [self.corpus.word(i)
                            for i in weightedWordIdx[:n_words]]

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


def topics_by_integer_property(lda, all_property_values, delta=5):
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
    figure(figsize=(16, 40))
    for idx, topic in enumerate(topicWeightedWords):
        wc = WordCloud(background_color="white")
        img = wc.generate_from_frequencies(
            [(word, weight) for weight, word in topic])
        subplot(len(topicWeightedWords), 2, 2 * idx + 1)
        imshow(img)
        axis('off')

        subplot(len(topicWeightedWords), 2, 2 * idx + 2)
        plot(topicsByProperty[:, idx])
        axis([10, 100, 0, 1.0])
        title('Topic #%2d' % (idx))


def getDocumentTopics(lda, corpus):
    docWeights = lda.transform(corpus.sparse_matrix())[0]
    return docWeights / docWeights.sum()
