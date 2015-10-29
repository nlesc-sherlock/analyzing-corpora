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

import argparse
from corpora import load_sparse_corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="load files of corpus and store sparse corpus")
    parser.add_argument(
        'corpus',
        help="python pickle file, containing tokens and metadata")
    parser.add_argument('matrix')
    parser.add_argument('-d', '--dictionary', default=None)
    parser.add_argument('-m', '--metadata', default=None)
    args = parser.parse_args()

    print("loading corpus")
    corpus = load_sparse_corpus(
        documents_file=args.corpus,
        dictionary_file=args.dictionary,
        metadata_file=args.metadata)

    print("writing matrix to file")
    corpus.save(sparse_matrix_file=args.matrix,
                metadata_filename=args.metadata_file)
