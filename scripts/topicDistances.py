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

from scipy.spatial.distance import cosine
from corpora.scikit import ScikitLda
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate distance between topics")
    parser.add_argument('-t', '--topic-model', default=None, required=True)
    args = parser.parse_args()

    lda = ScikitLda.load(args.topic_model)

    topics = []
    for topic in lda.topics:
        topics.append(topic / topic.sum())

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            print 'Topic#%d Topic#%d %f' % (i, j, cosine(topics[i], topics[j]))
