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
import os
import gensim
import argparse
from corpora.corpus import Corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train and save the LDA parser with different number of topics")
    parser.add_argument('output_folder')
    parser.add_argument('-p', '--parsed-document',
        help="python pickle file, containing tokens and metadata", default=None)
    parser.add_argument('-d', '--dictionary', default=None)
    parser.add_argument('-a', '--no-above', default=0.5)
    parser.add_argument('-b', '--no-below', default=5)
    parser.add_argument('-k', '--keep', default=1000000)
    args = parser.parse_args()

    if args.dictionary is None:
        if args.parsed_document is None:
            raise ValueError("Need parsed document or dictionary")
        dic = None
    else:
        print("loading dictionary")
        dic = gensim.corpora.Dictionary.load(args.dictionary)

    if args.parsed_document is None:
        tokens = None
        metadata = None
    else:
        print("loading pickled data")
        with open(args.parsed_document) as f:
            data = pickle.load(f)
        tokens = data['tokens']
        metadata = data['metadata']

    corpus = Corpus(documents=tokens, metadata=metadata, dictionary=dic)
    corpus.filter_extremes(no_above=args.no_above, no_below=args.no_below, keep_n=args.keep)

    print("writing to file")
    output_file = os.path.join(args.output_folder,
                               'filtered_{0}_{1}_{2}.dic'.format(args.no_above,
                                                                 args.no_below,
                                                                 args.keep_n))

    corpus.save_dictionary(output_file)
