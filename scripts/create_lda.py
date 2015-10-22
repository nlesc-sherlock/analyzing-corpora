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

import pickle
import joblib
import os
import gensim
from corpora.corpus import Corpus
from corpora.scikit import ScikitLda

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train and save the LDA parser with different number of topics")
    parser.add_argument('output_folder')
    parser.add_argument(
        'parsed_document',
        help="python pickle file, containing tokens and metadata")
    parser.add_argument('-d', '--dictionary', default=None)
    parser.add_argument('-t', '--topics', default=10)
    args = parser.parse_args()

    if args.dictionary is None:
        dic = None
    else:
        print("loading dictionary")
        dic = gensim.corpora.Dictionary.load(filename)

    print("loading pickled data")
    with open(args.parsed_document) as f:
        data = pickle.load(f)

    corpus = Corpus(documents=data['tokens'], metadata=data['metadata'],
                    dictionary=dic)

    print("calculating LDA")
    lda = ScikitLda(corpus, n_topics=args.topics)

    output_file = os.path.join(args.output_folder,
                               'lda_{0}.pkl'.format(args.topics))
    joblib.dump(lda.lda, output_file)
