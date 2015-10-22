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

from __future__ import print_function
from corpora.scikit import (ScikitLda, plot_wordcloud_with_property,
                            topics_by_integer_property,
                            topics_by_discrete_property)
from corpora.corpus import load_vraagtekst_corpus
from corpora.util import select_top
from matplotlib.pyplot import savefig
import numpy as np

if __name__ == '__main__':
    corpus = load_vraagtekst_corpus('data/preprocessedData.pkl')

    print("nSamples (docs) : {0}".format(corpus.num_samples))
    print("nFeatures(words): {0}".format(corpus.num_features))

    lda = ScikitLda(corpus, n_topics=10)
    topicWords, topicWeightedWords = lda.topic_words()

    for topic_idx, wordsInTopic in enumerate(topicWords):
        print("Topic #{0}:".format(topic_idx))
        print(" ".join(wordsInTopic))

    topicsByOrg, orgs = topics_by_discrete_property(
        lda, corpus.metadata_frame['individu of groep'])
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

    topicsByAge = topics_by_integer_property(
        lda, corpus.metadata_frame['Leeftijd'])
    plot_wordcloud_with_property(topicWeightedWords, topicsByAge)
    savefig('topics.png')
