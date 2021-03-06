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
from sklearn.externals import joblib


class ScikitLda(object):

    def __init__(self, corpus=None, lda=None, n_topics=10,
                 max_iter=5, learning_method='online', learning_offset=50.,
                 **kwargs):
        if lda is None:
            self.lda = LatentDirichletAllocation(
                n_topics=n_topics, max_iter=max_iter,
                learning_method=learning_method,
                learning_offset=learning_offset, **kwargs)
        else:
            self.lda = lda

        self._corpus = corpus
        self._weights = None

    def fit_transform(self):
        return self.lda.fit_transform(self.corpus.sparse_matrix())

    def fit(self):
        self.lda.fit(self.corpus.sparse_matrix())

    def partial_fit(self, corpus):
        self.lda.partial_fit(corpus.sparse_matrix())
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
            self._weights = self.partial_weights(self.corpus)
        return self._weights

    def partial_weights(self, corpus):
        weights = self.transform(corpus)
        return (weights.T / weights.sum(axis=1)).T

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

    def save(self, filename):
        joblib.dump(self.lda, filename)

    @classmethod
    def load(cls, filename, corpus=None):
        lda = joblib.load(filename)
        return cls(lda=lda, corpus=corpus)


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
